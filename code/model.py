import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from mytools.tool_utils import StringUtils, FileUtils, logging
import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss, LayerNorm


model_registry = {}


def register_model(cls):
    snake_case_name = StringUtils.camel_to_snake(cls.__name__)
    model_registry[snake_case_name] = cls
    return cls


def create_model(model_name: str, model_args: dict):
    model_name = model_name if model_name.endswith("_model") else model_name.strip() + "_model"
    model = model_registry[model_name](model_args)
    return model

def get_model_class(model_name: str):
    model_name = model_name if model_name.endswith("_model") else model_name.strip() + "_model"
    return model_registry[model_name] if model_name in model_registry else None


@register_model
class RetrievalModel(nn.Module):

    def __init__(self, args):
        super(RetrievalModel, self).__init__()
        pretrained_dir = args['model_config_dir']
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
        self.config = AutoConfig.from_pretrained(pretrained_dir)
        self.repr_layer_index = args.get("repr_layer_index", self.config.num_hidden_layers)
        logging.info("Extract representations from {}".format(self.repr_layer_index))
        self.pad_id = self.config.pad_token_id
        init_from_hf = args.get("init_from_hf", False)
        self.enable_tok_loss = args.get("enable_tok_loss", False)
        self.enable_sent_loss = args.get("enable_sent_loss", False)
        self.enable_cycle_loss = args.get("enable_cycle_loss", False)
        self.tok_loss_weight = args.get("tok_loss_weight", 1.0)
        self.sent_loss_weight = args.get("sent_loss_weight", 1.0)
        self.cycle_loss_weight = args.get("cycle_loss_weight", 1.0)
        self.enable_tokenizer_head = args.get("enable_tokenizer_head", False)
        dropout_rate = args.get("dropout_rate", None)
        if dropout_rate is not None:
            for k, v in self.config.__dict__.items():
                if "dropout" in k and v is not None and dropout_rate != v:
                    self.config.__dict__[k] = dropout_rate
                    logging.info("Reset `{}={}` to {}".format(k, v, dropout_rate))
        logging.info("Init from huggingface is {}".format(init_from_hf))
        if self.enable_tok_loss or self.enable_sent_loss:
            self.encoder_model = AutoModel.from_config(self.config, add_pooling_layer=True)
        else:
            self.encoder_model = AutoModel.from_config(self.config, add_pooling_layer=False)
        if init_from_hf:
            logging.info("Init by the huggingface model parameters")
            updated_state_dict = dict()
            if FileUtils.exists("{}/pytorch_model.bin".format(pretrained_dir)):
                ckpt_state_dict = FileUtils.load_file("{}/pytorch_model.bin".format(pretrained_dir))
            elif FileUtils.exists("{}/model.safetensors".format(pretrained_dir)):
                ckpt_state_dict = FileUtils.load_file("{}/model.safetensors".format(pretrained_dir))
            else:
                raise ValueError("Cannot find ckpt file under {}".format(args['model_pretrained_dir']))
            for k in self.encoder_model.state_dict().keys():
                updated_state_dict[k] = ckpt_state_dict[k]
            self.encoder_model.load_state_dict(updated_state_dict)
        self.hidden_size = self.config.hidden_size
        if args['out_hidden_size'] > 0:
            self.linear = nn.Linear(self.hidden_size * 2, args['out_hidden_size'])
        if self.enable_tokenizer_head:
            self.tkz_loss_weight = args.get("tkz_loss_weight", 1.0)
            self.tokenizer_head = nn.Linear(self.hidden_size * 2, 2)
            self.tokenizer_criterion = nn.CrossEntropyLoss()
        self.version = args.get("version", "simcse_tkz")
        self.norm_repr = args.get('norm_repr', False)
        logging.info("Enable repr normalization: {}".format(self.norm_repr))
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        
    @classmethod
    def from_pretrained(cls, args):
        logging.info("Initializing model from pre-trained")
        model = cls(args)
        if FileUtils.exists(args['model_pretrained_dir'] + "/pytorch_model.bin"):
            state_dict = FileUtils.load_from_disk(args['model_pretrained_dir'] + "/pytorch_model.bin", 'pt')
        elif FileUtils.exists(args['model_pretrained_dir'] + "/model.safetensors"):
            state_dict = FileUtils.load_from_disk(args['model_pretrained_dir'] + "/model.safetensors", 'safetensors')
        else:
            raise ValueError("Cannot find ckpt file under {}".format(args['model_pretrained_dir']))

        updated_state_dict = {}
        for k in model.state_dict().keys():
            updated_state_dict[k] = state_dict[k]
        for k in state_dict.keys():
            if k not in updated_state_dict:
                logging.info("parameters of {} in the ckpt was not used".format(k))

        model.load_state_dict(updated_state_dict)
        return model

    def save_pretrained(self, save_dir):
        logging.info("Saving model to {}".format(save_dir))
        FileUtils.check_dirs(save_dir)
        # self.encoder_tokenizer.save_pretrained(save_dir)
        # self.encoder_model.save_pretrained(save_dir)
        state_dict = self.state_dict()
        FileUtils.save_to_disk(state_dict, save_dir + "pytorch_model.bin", 'pt')
    
    def forward(self, batch):
        loss = self.forward_simplified_with_tokenization(batch)
        return loss

    def forward_simplified_with_tokenization(self, batch):
        src_out = self.encoder_model(input_ids=batch['src_input_ids'], attention_mask=batch['src_attention_mask'], output_hidden_states=True)
        ref_out = self.encoder_model(input_ids=batch['ref_input_ids'], attention_mask=batch['ref_attention_mask'], output_hidden_states=True)
        src_hidden_states = src_out.hidden_states[self.repr_layer_index]
        ref_hidden_states = ref_out.hidden_states[self.repr_layer_index]
        src_phrase_hidden_states_start = src_hidden_states[batch['src_phrase_align_data'][0], batch['src_phrase_align_data'][1]]
        src_phrase_hidden_states_end = src_hidden_states[batch['src_phrase_align_data'][0], batch['src_phrase_align_data'][2]]
        src_phrase_hidden_states = self.linear(torch.cat([src_phrase_hidden_states_start, src_phrase_hidden_states_end], dim=-1))

        ref_pharse_hidden_states_start = ref_hidden_states[batch['ref_phrase_align_data'][0], batch['ref_phrase_align_data'][1]]
        ref_pharse_hidden_states_end = ref_hidden_states[batch['ref_phrase_align_data'][0], batch['ref_phrase_align_data'][2]]
        ref_phrase_hidden_states = self.linear(torch.cat([ref_pharse_hidden_states_start, ref_pharse_hidden_states_end], dim=-1))

        s2t_num, t2s_num = batch['s2t_align_labels'].size(0), batch['t2s_align_labels'].size(0)
        s2t_align_logits = torch.matmul(src_phrase_hidden_states[:s2t_num], torch.cat([ref_phrase_hidden_states, src_phrase_hidden_states[s2t_num:]], dim=0).transpose(0, 1))
        t2s_align_logits = torch.matmul(ref_phrase_hidden_states[:t2s_num], torch.cat([src_phrase_hidden_states, ref_phrase_hidden_states[t2s_num:]], dim=0).transpose(0, 1))
        
        s2t_loss = self.criterion(s2t_align_logits, batch['s2t_align_labels'])
        t2s_loss = self.criterion(t2s_align_logits, batch['t2s_align_labels'])

        src_tkz_loss = self.step_tokenize_loss(
            src_hidden_states, batch['src_phrase_tkz_data'][0], batch['src_phrase_tkz_data'][1],
            batch['src_phrase_tkz_data'][2], batch['src_phrase_tkz_labels'])
        ref_tkz_loss = self.step_tokenize_loss(
            ref_hidden_states, batch['ref_phrase_tkz_data'][0], batch['ref_phrase_tkz_data'][1],
            batch['ref_phrase_tkz_data'][2], batch['ref_phrase_tkz_labels'])
        return s2t_loss + t2s_loss + self.tkz_loss_weight * (ref_tkz_loss + src_tkz_loss)


    def step_tokenize_loss(self, hidden_states, phrase_tkz_batch_ids, phrase_tkz_start_ids, phrase_tkz_end_ids, phrase_tkz_labels):
        hidden_states_start = hidden_states[phrase_tkz_batch_ids, phrase_tkz_start_ids]
        hidden_states_end = hidden_states[phrase_tkz_batch_ids, phrase_tkz_end_ids]
        logits = self.tokenizer_head(torch.cat([hidden_states_start, hidden_states_end], dim=-1))
        loss = self.tokenizer_criterion(logits, phrase_tkz_labels)
        return loss

    def step_alignment_loss(self, src_input_ids, src_attention_mask,
                  src_phrase_batch_ids, src_phrase_start_ids, src_phrase_end_ids,
                  s2t_input_ids, s2t_attention_mask, 
                  s2t_phrase_batch_ids, s2t_phrase_start_ids, s2t_phrase_end_ids, 
                  ref_input_ids, ref_attention_mask, s2t_alignment_labels, 
                  src_out=None, ref_out=None, return_hidden_states=False):

        hidden_states = {"src_out": None, "ref_out": None}
        if src_out is None:
            src_out = self.encoder_model(input_ids=src_input_ids, attention_mask=src_attention_mask, output_hidden_states=True)
            hidden_states['src_out'] = src_out
        elif return_hidden_states:
            hidden_states['src_out'] = src_out
        s2t_out = self.encoder_model(input_ids=s2t_input_ids, attention_mask=s2t_attention_mask, output_hidden_states=True)
        if self.enable_sent_loss or self.enable_sent_loss:
            if ref_out is None:
                ref_out = self.encoder_model(input_ids=ref_input_ids, attention_mask=ref_attention_mask, output_hidden_states=True)
                hidden_states['ref_out'] = ref_out
            elif return_hidden_states:
                hidden_states['ref_out'] = ref_out
        src_hidden_states = src_out.hidden_states[self.repr_layer_index]
        s2t_hidden_sattes = s2t_out.hidden_states[self.repr_layer_index]
        src_phrase_hidden_states_start = src_hidden_states[src_phrase_batch_ids, src_phrase_start_ids]
        src_phrase_hidden_states_end = src_hidden_states[src_phrase_batch_ids, src_phrase_end_ids]
        src_phrase_hidden_states = self.linear(torch.cat([src_phrase_hidden_states_start,src_phrase_hidden_states_end], dim=-1))

        s2t_pharse_hidden_states_start = s2t_hidden_sattes[s2t_phrase_batch_ids, s2t_phrase_start_ids]
        s2t_pharse_hidden_states_end = s2t_hidden_sattes[s2t_phrase_batch_ids, s2t_phrase_end_ids]
        s2t_phrase_hidden_states = self.linear(torch.cat([s2t_pharse_hidden_states_start,s2t_pharse_hidden_states_end], dim=-1))
        
        if self.norm_repr:
            src_phrase_hidden_states_norm = torch.norm(src_phrase_hidden_states, dim=-1, keepdim=True)
            s2t_phrase_hidden_states_norm = torch.norm(s2t_phrase_hidden_states, dim=-1, keepdim=True)
            src_phrase_hidden_states = src_phrase_hidden_states / src_phrase_hidden_states_norm
            s2t_phrase_hidden_states = s2t_phrase_hidden_states / s2t_phrase_hidden_states_norm

        logits = torch.matmul(src_phrase_hidden_states, s2t_phrase_hidden_states.transpose(0, 1))
        labels = s2t_alignment_labels

        loss = self.criterion(logits, labels)
        if self.enable_sent_loss:
            batch_size = src_out.pooler_output.size(0)
            candidates = torch.cat([ref_out.pooler_output, s2t_out.pooler_output], dim=0)
            logits_sent = torch.matmul(src_out.pooler_output, candidates.t())
            prob_sent = - F.log_softmax(logits_sent, dim=-1)
            loss_sent = torch.diag(prob_sent).sum() / batch_size
            loss = loss + self.sent_loss_weight * loss_sent
        
        if self.enable_tok_loss:
            fw_prob = - F.log_softmax(torch.matmul(src_out.pooler_output, self.encoder_model.embeddings.word_embeddings.weight.t()), dim=-1)
            loss_tok_fw = torch.sum(torch.gather(fw_prob, -1, ref_input_ids) * ref_attention_mask) / torch.sum(ref_attention_mask)
            loss = loss + self.tok_loss_weight * loss_tok_fw

            if self.enable_cycle_loss:
                loss_cycle_src = torch.sum(torch.gather(fw_prob, -1, src_input_ids) * src_attention_mask) / torch.sum(src_attention_mask)
                loss = loss + self.cycle_loss_weight * loss_cycle_src
        if return_hidden_states:
            return loss, hidden_states
        else:
            return loss
    
    def encode_batch(self, batch, do_tokenize=False, verbose=False, **kwargs):
        with torch.no_grad():
            out = self.encoder_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            hidden_states = out.hidden_states[self.repr_layer_index]
            if do_tokenize:
                phrase_batch_ids, phrase_start_ids, phrase_end_ids = self.tokenize(batch, hidden_states, **kwargs)
            else:
                phrase_batch_ids, phrase_start_ids, phrase_end_ids = batch['phrase_batch_ids'], batch['phrase_start_ids'], batch['phrase_end_ids']
            phrase_hidden_states_start = hidden_states[phrase_batch_ids, phrase_start_ids]
            phrase_hidden_states_end = hidden_states[phrase_batch_ids, phrase_end_ids]
            phrase_hidden_states = self.linear(torch.cat([phrase_hidden_states_start, phrase_hidden_states_end], dim=-1))

            if self.norm_repr:
                phrase_hidden_states_norm = torch.norm(phrase_hidden_states, dim=-1, keepdim=True)
                phrase_hidden_states = phrase_hidden_states / phrase_hidden_states_norm
        if verbose:
            return phrase_hidden_states, {"phrase_batch_ids": phrase_batch_ids, "phrase_start_ids": phrase_start_ids, "phrase_end_ids": phrase_end_ids}
        else:
            return phrase_hidden_states

    def _generate_combinations(self, sequence):
        # Generating combinations for the sequence
        seq_len = len(sequence)
        seq_indices = torch.arange(1, seq_len-1, device=sequence.device)
        # comb = torch.combinations(sequence, r=2, with_replacement=True)
        indices_comb = torch.combinations(seq_indices, r=2, with_replacement=True)
        return indices_comb

    def tokenize(self, batch, hidden_states=None, max_phrase_len=6, threshold=0.5):
        if hidden_states is None:
            out = self.encoder_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            hidden_states = out.hidden_states[self.repr_layer_index]
        phrase_batch_ids, phrase_start_ids, phrase_end_ids = None, None, None
        batch_size = hidden_states.size(0)
        for batch_idx in range(batch_size):
            real_len = batch['attention_mask'][batch_idx].sum()
            sequence = batch['input_ids'][batch_idx]
            phrase_indices = self._generate_combinations(sequence[:real_len])
            tkz_phrase_start_ids, tkz_phrase_end_ids = phrase_indices[:, 0], phrase_indices[:, 1]
            tkz_phrase_batch_ids = torch.zeros_like(tkz_phrase_start_ids) + batch_idx
            if phrase_batch_ids is None:
                phrase_batch_ids = tkz_phrase_batch_ids
                phrase_start_ids = tkz_phrase_start_ids
                phrase_end_ids = tkz_phrase_end_ids
            else:
                phrase_batch_ids = torch.cat([phrase_batch_ids, tkz_phrase_batch_ids], dim=-1)
                phrase_start_ids = torch.cat([phrase_start_ids, tkz_phrase_start_ids], dim=-1)
                phrase_end_ids = torch.cat([phrase_end_ids, tkz_phrase_end_ids], dim=-1)
        overlength_indicator = (phrase_end_ids + 1 - phrase_start_ids) <= max_phrase_len
        phrase_batch_ids = phrase_batch_ids[overlength_indicator]
        phrase_start_ids = phrase_start_ids[overlength_indicator]
        phrase_end_ids = phrase_end_ids[overlength_indicator]
        hidden_states_start = hidden_states[phrase_batch_ids, phrase_start_ids]
        hidden_states_end = hidden_states[phrase_batch_ids, phrase_end_ids]
        tokenization_logits = self.tokenizer_head(torch.cat([hidden_states_start,hidden_states_end], dim=-1))
        phrase_score = F.softmax(tokenization_logits, dim=-1)[:, 1]
        phrase_indicator = phrase_score > threshold
        return phrase_batch_ids[phrase_indicator], phrase_start_ids[phrase_indicator], phrase_end_ids[phrase_indicator]


def _test_retrival_model():
    from dataloader import create_dataloader
    from collections import defaultdict
    from mytools.tool_utils import TorchUtils
    device = torch.device("cuda")
    torch.manual_seed(10086)
    random.seed(10086)
    np.random.seed(10086)
    DEST = "/apdcephfs/share_916081/redmondliu/intern_data/alanili/Copyisallyouneed/data/wmt16_deen/"
    ds_name = "retrieval_dataset"
    ds_args = {
        "shuffle": True,
        "batch_size": 4,
        "src_phrase_tokenizer_path": DEST + "/320k.lp.mpl36.bpe.de.model.model",
        "tgt_phrase_tokenizer_path": DEST + "/320k.lp.mpl36.bpe.en.model.model",
        "src_file": DEST + "/debug.clean.de",
        "tgt_file": DEST + "/debug.clean.en",
        "phrase_table_path": DEST + "/train.clean.phtab.json",
        "encoder_tokenizer_path": "/apdcephfs/share_916081/redmondliu/intern_data/alanili/huggingface/xlm-roberta-base/",
        "min_phrase_freq": 5,
        'negative_num': 5,
        'from_pretrained': False
    }
    debug_dataloader = create_dataloader(ds_name, ds_args)
    model_name = "retrieval"
    # model_args = {
    #     "pretrained_dir": "/apdcephfs/share_916081/redmondliu/intern_data/alanili/huggingface/xlm-roberta-base",
    #     "encoder_layer_index": 12,
    #     "out_hidden_size": 128,
    #     "save_model_to": "/apdcephfs/share_916081/redmondliu/intern_data/alanili/reinvent_pbmt/ckpts/debug_model"
    # }
    # model = create_model(model_name, model_args)
    # model = model.to(device)
    # model = model.train()
    # for batch in debug_dataloader:
    #     batch = TorchUtils.move_to_device(batch, device)
    #     loss = model(batch)
    #     print(loss)
    #     break
    # save_model_to = "/apdcephfs/share_916081/redmondliu/intern_data/alanili/reinvent_pbmt/ckpts/debug_model"
    # model.save_pretrained(save_model_to)
    model_args = {
        "pretrained_dir": "/apdcephfs/share_916081/redmondliu/intern_data/alanili/reinvent_pbmt/ckpts/debug_model",
        "encoder_layer_index": 12,
        "out_hidden_size": 128,
    }
    model = create_model(model_name, model_args)
    model = model.to(device)
    for batch in debug_dataloader:
        batch = TorchUtils.move_to_device(batch, device)
        loss = model(batch)
        print(loss)
        break


if __name__ == "__main__":
    _test_retrival_model()