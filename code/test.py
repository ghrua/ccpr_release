import sentencepiece as spm
import fire
import json
import os
import os.path as path
from tqdm import tqdm
import torch
from collections import Counter
from mytools.tool_utils import FileUtils, DataUtils, MPUtils, HFUtils, FileType, LANGUAGE_LIST, TorchUtils, logging
from mytools.faiss_tools import build_index, search_single_file, search_files
from transformers import AutoModel, AutoTokenizer
from model import get_model_class
from dataloader import get_dataset_class
from sentence_transformers import SentenceTransformer
from common_utils import override

spm.set_random_generator_seed(10086) # ensure the tokenization is re-producible

class ModelType:
    OURS = "ours" # model trained by us
    HF = "hf" # huggingface transformers
    XPR = "xpr" # xpr model https://huggingface.co/cwszz/XPR/


class PhraseExtractionMethod:
    SUB = "sub" # substraction
    MEAN = "mean" # mean
    CAT = "cat" # concatenation


def _fix_shard_repr_index(shard_idx_fpath, shard_start_idx):
    idx_data = FileUtils.load_file(shard_idx_fpath)
    for i in range(len(idx_data)):
        idx_data[i] += shard_start_idx
    FileUtils.save_file(idx_data, shard_idx_fpath)
    return len(idx_data)


def _hf_model_encode_batch(model, batch, layer_index=-2, extraction_method=PhraseExtractionMethod.CAT):
    with torch.no_grad():
        out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        hidden_states = out.hidden_states[layer_index]
        phrase_hidden_states_start = hidden_states[batch['phrase_batch_ids'], batch['phrase_start_ids']]
        phrase_hidden_states_end = hidden_states[batch['phrase_batch_ids'], batch['phrase_end_ids']]
        if extraction_method == PhraseExtractionMethod.CAT:
            phrase_hidden_states = torch.cat([phrase_hidden_states_start, phrase_hidden_states_end], dim=-1)
        elif extraction_method == PhraseExtractionMethod.SUB:
            phrase_hidden_states = phrase_hidden_states_end - phrase_hidden_states_start
        elif extraction_method == PhraseExtractionMethod.MEAN:
            phrase_hidden_states = (phrase_hidden_states_end + phrase_hidden_states_start) / 2
        else:
            raise NotImplementedError("Unkown PhraseExtractionMethod: {}".format(extraction_method))
    return phrase_hidden_states


def _encode_data(args, data, device, proc_id, sent_start_id, prefix="phrase", encode_lang=None, model_type=ModelType.OURS, encoder_layer_index=-1, extraction_method=PhraseExtractionMethod.CAT):
    """
    NOTE: the subprocess may finish before the files are well saved on disk. Therefore, please run the encoding process and indexing process seperately.
    """
    logging.info("Proc-{} | Processing {} data examples".format(proc_id, len(data)))
    logging.info("Proc-{} | The used device is {}".format(proc_id, device))
    logging.info("Proc-{} | The sent id starts from {}".format(proc_id, sent_start_id))

    args['model_pretrained_dir'] = args['inference_model_dir']
    # overide args that are provided from command line:
    if 'inference_with_ner' in args and args['inference_with_ner']:
        import stanza
        ner_model = stanza.Pipeline(lang=args['tgt_lang'], processors='tokenize,ner', model_dir="./cache/", tokenize_pretokenized=True)
    if model_type == ModelType.OURS:
        model = get_model_class(args['model_name']).from_pretrained(args)
        encoder_tokenizer = AutoTokenizer.from_pretrained(args['model_config_dir'])
    elif model_type == ModelType.HF:
        model = AutoModel.from_pretrained(args['model_config_dir'])
        encoder_tokenizer = AutoTokenizer.from_pretrained(args['model_config_dir'])
    elif model_type == ModelType.XPR:
        pass
    else:
        raise NotImplementedError("Unknown model name: {}".format(args['model_name']))
    
    model = model.eval().to(device)
    dataset_cls = get_dataset_class(args['ds_name'])
    if 'mt_subword_tokenizer_path' in args:
        mt_subword_tokenizer = DataUtils.load_spm(args['mt_subword_tokenizer_path']) 
    else:
        mt_subword_tokenizer = None
    phrase_tokenizer = None
    moses_tokenizer = DataUtils.get_moses_tokenizer(encode_lang)
    # if encode_lang is None or encode_lang == args['tgt_lang']:
    #     if 'tgt_phrase_tokenizer_path' in args:
    #         phrase_tokenizer = DataUtils.load_spm(args['tgt_phrase_tokenizer_path'])
    #     else:
    #         phrase_tokenizer = None
    # elif encode_lang == args['src_lang']:
    #     if 'src_phrase_tokenizer_path' in args:
    #         phrase_tokenizer = DataUtils.load_spm(args['src_phrase_tokenizer_path'])
    #     else:
    #         phrase_tokenizer = None
    # else:
    #     raise ValueError("Unknow language to encode: {}".format(encode_lang))

    batch_sents, batch_sent_ids, indices, repr_list, phrase_list, sentid_list = [], [], [], [], [], []
    phrase_start_ids_list, phrase_end_ids_list = [], []
    n_repr, n_shard, batch_size = 0, 0, args['inference_batch_size']
    FileUtils.check_dirs(FileUtils.get_dir(args['inference_save_dir']))
    if args['inference_data_mode'] == "ngram":
        # ngram_sets = FileUtils.load_file(args['inference_tgt_ngram_set'])
        ngram_sets = None
    # n_shard = 43
    inference_tokenizer_max_phrase_len = args.get('inference_tokenizer_max_phrase_len', 6)
    inference_tokenizer_threshold = args.get('inference_tokenizer_threshold', 0.5)
    for sid, line in enumerate(data):
        if (len(line.strip().split()) >= args['max_sequence_len']):
            continue
        batch_sents.append(line.strip())
        batch_sent_ids.append(sid+sent_start_id)
        if len(batch_sents) >= batch_size:
            batch = encoder_tokenizer(batch_sents, truncation=True, padding=True, return_tensors="pt")           
            if args['inference_data_mode'] == "ngram":
                aux_batch = dataset_cls.prepare_ngram_inference_batch(ngram_sets, batch_sents, batch_sent_ids, encoder_tokenizer, moses_tokenizer=moses_tokenizer, max_ngram_len=args['inference_max_ngram_len'])
            elif args['inference_data_mode'] == "tokenizer":
                aux_batch = dataset_cls.prepare_tokenizer_inference_batch(batch_sents, batch_sent_ids, encoder_tokenizer, phrase_tokenizer, mt_subword_tokenizer, only_phrase=False, n_sampling=args['inference_tokenizer_n_sampling'])
            elif args['inference_data_mode'] != "model":
                raise ValueError
            batch = TorchUtils.move_to_device(batch, device=device)
            if model_type == ModelType.OURS:
                if args['inference_data_mode'] == "model":
                    phrase_hidden_states, aux_batch = model.encode_batch(batch, do_tokenize=True, verbose=True, max_phrase_len=inference_tokenizer_max_phrase_len, threshold=inference_tokenizer_threshold)
                    aux_batch['sent_ids'] = [batch_sent_ids[bid] for bid in aux_batch['phrase_batch_ids']]
                    aux_batch['phrase_piece'] = []
                    for bid, start_id, end_id in zip(aux_batch['phrase_batch_ids'], aux_batch['phrase_start_ids'], aux_batch['phrase_end_ids']):
                        aux_batch['phrase_piece'].append(" ".join(encoder_tokenizer.convert_ids_to_tokens(batch['input_ids'][bid, start_id:end_id+1])))
                else:
                    batch.update(aux_batch)
                    batch.pop("phrase_piece")
                    phrase_hidden_states = model.encode_batch(batch)
            elif model_type ==  ModelType.HF:
                phrase_hidden_states = _hf_model_encode_batch(model, batch, encoder_layer_index, extraction_method)
            else:
                raise NotImplementedError("Unknow encoding type {}".format(model_type))
            phrase_start_ids = aux_batch['phrase_start_ids'].tolist()
            phrase_end_ids = aux_batch['phrase_end_ids'].tolist()
            phrase_piece = aux_batch.pop("phrase_piece")
            batch.update(aux_batch)
            sent_ids = aux_batch['sent_ids'].tolist() if not isinstance(aux_batch['sent_ids'], list) else aux_batch['sent_ids']
            for s_id, p_p, p_h, start_id, end_id in zip(sent_ids, phrase_piece, phrase_hidden_states.cpu(), phrase_start_ids, phrase_end_ids):
                indices.append(n_repr)
                repr_list.append(p_h)
                phrase_list.append(p_p)
                sentid_list.append(s_id)
                phrase_start_ids_list.append(start_id)
                phrase_end_ids_list.append(end_id)
                n_repr += 1

            batch_sents = []
            batch_sent_ids = []
            if len(indices) >= args['inference_cache_size']:
                FileUtils.save_to_disk(indices, args['inference_save_dir'] + "/{}.{}.{}.{}.repr.idx".format(prefix, encode_lang, proc_id, n_shard))
                FileUtils.save_to_disk(torch.stack(repr_list, dim=0), args['inference_save_dir'] + "/{}.{}.{}.{}.repr.dat".format(prefix, encode_lang, proc_id, n_shard))
                FileUtils.save_to_disk(phrase_list, args['inference_save_dir'] + "/{}.{}.{}.{}.repr.txt".format(prefix, encode_lang, proc_id, n_shard))
                FileUtils.save_to_disk(sentid_list, args['inference_save_dir'] + "/{}.{}.{}.{}.repr.sid".format(prefix, encode_lang, proc_id, n_shard))
                FileUtils.save_to_disk(phrase_start_ids_list, args['inference_save_dir'] + "/{}.{}.{}.{}.repr.spos".format(prefix, encode_lang, proc_id, n_shard))
                FileUtils.save_to_disk(phrase_end_ids_list, args['inference_save_dir'] + "/{}.{}.{}.{}.repr.epos".format(prefix, encode_lang, proc_id, n_shard))
                indices = []
                repr_list = []
                phrase_list = []
                sentid_list = []
                phrase_start_ids_list = []
                phrase_end_ids_list = []
                n_shard += 1
        
    if batch_sents:
        batch = encoder_tokenizer(batch_sents, truncation=True, padding=True, return_tensors="pt")
        if args['inference_data_mode'] == "ngram":
            aux_batch = dataset_cls.prepare_ngram_inference_batch(ngram_sets, batch_sents, batch_sent_ids, encoder_tokenizer, moses_tokenizer=moses_tokenizer, max_ngram_len=args['inference_max_ngram_len'])
        elif args['inference_data_mode'] == "tokenizer":
            aux_batch = dataset_cls.prepare_tokenizer_inference_batch(batch_sents, batch_sent_ids, encoder_tokenizer, phrase_tokenizer, mt_subword_tokenizer, only_phrase=False, n_sampling=args['inference_tokenizer_n_sampling'])
        elif args['inference_data_mode'] != "model":
            raise ValueError

        batch = TorchUtils.move_to_device(batch, device=device)
        if model_type == ModelType.OURS:
            if args['inference_data_mode'] == "model":
                phrase_hidden_states, aux_batch = model.encode_batch(batch, do_tokenize=True, verbose=True, max_phrase_len=args.get('inference_tokenizer_max_phrase_len', 6))
                aux_batch['sent_ids'] = [batch_sent_ids[bid] for bid in aux_batch['phrase_batch_ids']]
                aux_batch['phrase_piece'] = []
                for bid, start_id, end_id in zip(aux_batch['phrase_batch_ids'], aux_batch['phrase_start_ids'], aux_batch['phrase_end_ids']):
                    aux_batch['phrase_piece'].append(" ".join(encoder_tokenizer.convert_ids_to_tokens(batch['input_ids'][bid, start_id:end_id+1])))
            else:
                batch.update(aux_batch)
                batch.pop("phrase_piece")
                phrase_hidden_states = model.encode_batch(batch)
        elif model_type ==  ModelType.HF:
            phrase_hidden_states = _hf_model_encode_batch(model, batch, encoder_layer_index, extraction_method)
        else:
            raise NotImplementedError("Unknow encoding type {}".format(model_type))
        phrase_start_ids = aux_batch['phrase_start_ids'].tolist()
        phrase_end_ids = aux_batch['phrase_end_ids'].tolist()
        phrase_piece = aux_batch.pop("phrase_piece")
        batch.update(aux_batch)
        sent_ids = aux_batch['sent_ids'].tolist() if not isinstance(aux_batch['sent_ids'], list) else aux_batch['sent_ids']
        for s_id, p_p, p_h, start_id, end_id in zip(sent_ids, phrase_piece, phrase_hidden_states.cpu(), phrase_start_ids, phrase_end_ids):
            indices.append(n_repr)
            repr_list.append(p_h)
            phrase_list.append(p_p)
            sentid_list.append(s_id)
            phrase_start_ids_list.append(start_id)
            phrase_end_ids_list.append(end_id)
            n_repr += 1

    if indices:
        FileUtils.save_to_disk(indices, args['inference_save_dir'] + "/{}.{}.{}.{}.repr.idx".format(prefix, encode_lang, proc_id, n_shard))
        FileUtils.save_to_disk(torch.stack(repr_list, dim=0), args['inference_save_dir'] + "/{}.{}.{}.{}.repr.dat".format(prefix, encode_lang, proc_id, n_shard))
        FileUtils.save_to_disk(phrase_list, args['inference_save_dir'] + "/{}.{}.{}.{}.repr.txt".format(prefix, encode_lang, proc_id, n_shard))
        FileUtils.save_to_disk(sentid_list, args['inference_save_dir'] + "/{}.{}.{}.{}.repr.sid".format(prefix, encode_lang, proc_id, n_shard))
        FileUtils.save_to_disk(phrase_start_ids_list, args['inference_save_dir'] + "/{}.{}.{}.{}.repr.spos".format(prefix, encode_lang, proc_id, n_shard))
        FileUtils.save_to_disk(phrase_end_ids_list, args['inference_save_dir'] + "/{}.{}.{}.{}.repr.epos".format(prefix, encode_lang, proc_id, n_shard))
        n_shard += 1


def build_inference_index(retriever_inference_config, **overide_args):
    args = FileUtils.load_from_disk(retriever_inference_config)
    args = override(args, overide_args)
    if isinstance(args['inference_index_type'], tuple):
        args['inference_index_type'] = ",".join(args['inference_index_type'])
    logging.info("args: \n{}".format(args))
    n_device = torch.cuda.device_count()
    prefix = args['inference_save_file_prefix'] if args['inference_save_file_prefix'] is not None else "phrase"
    encode_lang = args['inference_encode_lang'] if args['inference_encode_lang'] is not None else args['tgt_lang']
    model_type = args['model_type'] if args.get('model_type', None) is not None else ModelType.OURS
    encoder_layer_index = args['encoder_layer_index'] if args.get('encoder_layer_index', None) is not None else -1
    encode_data = not args['data_no_encoding'] if args.get('data_no_encoding', None) is not None else True
    rename_data = not args['data_no_rename'] if args.get('data_no_rename', None) is not None else True
    phrase_extraction_method = args['phrase_extraction_method'] if args.get('phrase_extraction_method', None) is not None else PhraseExtractionMethod.MEAN
    if encode_data:
        data = FileUtils.load_file(args['inference_datastore_file'])
        shard_size  = (len(data) + n_device - 1) // n_device
        args_list, sent_start_id = [], 0
        for proc_id in range(n_device):
            shard_data = data[proc_id*shard_size:(proc_id+1)*shard_size]
            device = torch.device("cuda:{}".format(proc_id)) if proc_id < n_device else torch.device("cpu")
            args_list.append((args, shard_data, device, proc_id, sent_start_id, prefix, encode_lang, model_type, encoder_layer_index, phrase_extraction_method))
            sent_start_id += len(shard_data)
        # _encode_data(*args_list[0])
        MPUtils.mp_func(_encode_data, args_list)

    if rename_data:
        n_shard, shard_repr_start_idx = 0, 0
        for proc_id in range(n_device):
            device_shard_id = 0
            total_idx = 0
            while True:
                fpath =  args['inference_save_dir'] + "/{}.{}.{}.{}.repr.idx".format(prefix, encode_lang, proc_id, device_shard_id)
                if FileUtils.exists(fpath):
                    total_idx += _fix_shard_repr_index(fpath, shard_repr_start_idx)
                    FileUtils.rename(args['inference_save_dir'] + "/{}.{}.{}.{}.repr.idx".format(prefix, encode_lang, proc_id, device_shard_id), args['inference_save_dir'] + "/{}.{}.{}.repr.idx".format(prefix, encode_lang, n_shard))
                    FileUtils.rename(args['inference_save_dir'] + "/{}.{}.{}.{}.repr.dat".format(prefix, encode_lang, proc_id, device_shard_id), args['inference_save_dir'] + "/{}.{}.{}.repr.dat".format(prefix, encode_lang, n_shard))
                    FileUtils.rename(args['inference_save_dir'] + "/{}.{}.{}.{}.repr.txt".format(prefix, encode_lang, proc_id, device_shard_id), args['inference_save_dir'] + "/{}.{}.{}.repr.txt".format(prefix, encode_lang, n_shard))
                    FileUtils.rename(args['inference_save_dir'] + "/{}.{}.{}.{}.repr.sid".format(prefix, encode_lang, proc_id, device_shard_id), args['inference_save_dir'] + "/{}.{}.{}.repr.sid".format(prefix, encode_lang, n_shard))
                    FileUtils.rename(args['inference_save_dir'] + "/{}.{}.{}.{}.repr.spos".format(prefix, encode_lang, proc_id, device_shard_id), args['inference_save_dir'] + "/{}.{}.{}.repr.spos".format(prefix, encode_lang, n_shard))
                    FileUtils.rename(args['inference_save_dir'] + "/{}.{}.{}.{}.repr.epos".format(prefix, encode_lang, proc_id, device_shard_id), args['inference_save_dir'] + "/{}.{}.{}.repr.epos".format(prefix, encode_lang, n_shard))
                    device_shard_id += 1
                    n_shard += 1
                else:
                    break
            shard_repr_start_idx += total_idx
            logging.info("New shard start idx: {}".format(shard_repr_start_idx))
    if args['inference_build_index']:
        from glob import glob
        dataprefix = args['inference_save_dir'] + "/{}.{}".format(prefix, encode_lang)
        n_data_shrads = len(list(glob(dataprefix + "*.repr.dat")))
        build_index(dataprefix=args['inference_save_dir'] + "/{}.{}".format(prefix, encode_lang),
                    index_dir=args['inference_save_dir'],
                    num_data_shards=n_data_shrads,
                    index_type=args['inference_index_type'],
                    num_index_shards=1, index_name_tag=args.get("inference_index_name_tag", ""))


def search_inference_index(retriever_inference_config, **overide_args):
    args = FileUtils.load_from_disk(retriever_inference_config)
    args = override(args, overide_args)
    if isinstance(args['inference_index_type'], tuple):
        args['inference_index_type'] = ",".join(args['inference_index_type'])
    logging.info("args: \n{}".format(args))
    prefix = args['inference_save_file_prefix'] if args['inference_save_file_prefix'] is not None else "phrase"
    encode_lang = args['inference_encode_lang'] if args['inference_encode_lang'] is not None else args['tgt_lang']
    queryprefix_list = []
    n_shard = 0
    while True:
        fpath =  args['inference_save_dir'] + "/{}.{}.{}.repr.idx".format(prefix, encode_lang, n_shard)
        if FileUtils.exists(fpath):
            queryprefix_list.append(args['inference_save_dir'] + "/{}.{}.{}".format(prefix, encode_lang, n_shard))
            n_shard += 1
        else:
            logging.info("Cannot find {}".format(fpath))
            break
    if args['inference_search_queryshard_ids']:
        queryprefix_list = [queryprefix_list[i] for i in args['inference_search_queryshard_ids']]
    index_name_tag = args.get("inference_index_name_tag", "")
    if index_name_tag:
        index_path = args['inference_save_dir'] + "/{}.{}.index".format(args['inference_index_type'], index_name_tag)
    else:
        index_path = args['inference_save_dir'] + "/{}.index".format(args['inference_index_type'])
    save_suffix = "{}.{}.retrieval.out.pt".format(args['inference_index_type'], index_name_tag) if index_name_tag else "retrieval.out.pt"
    search_files(queryprefix_list=queryprefix_list,
                index_path=index_path,
                save_suffix=save_suffix,
                search_batch_size=args['inference_search_batch_size'], index_on_gpu=args['inference_on_gpu'],
                search_topk=args['inference_retrieval_topk'], gpu_num=args['inference_gpu_num'])





if __name__ == "__main__":
    fire.Fire({
        "build_inference_index": build_inference_index,
        "search_inference_index": search_inference_index
    })
