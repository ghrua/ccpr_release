import sentencepiece as spm
import fire
import json
import os
import os.path as path
from PIL import Image
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import torch
from mytools.tool_utils import FileUtils, HFUtils, MPUtils
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)

MM_MODEL_TYPE_LIST = ["clip"]
MODALITY_LIST = ['text', 'vision', 'all']
MODE_LIST = ["cls", "avg"]
CLS_POS_DICT= {
    "bert": 0, "gpt2": -1, "clip": {'text': -1, 'vision': 0}, "roberta": 0
}

class ExtractionConfig:
    is_mm = False
    model_type = "bert"
    modality = None
    mode = "cls"
    layer = 12

    @staticmethod
    def show():
        logging.info("ExtractionConfig Information\nis_mm: {}\tmodel_type: {}\tmodality: {}\tmode: {}\tlayer: {}".format(
            ExtractionConfig.is_mm, ExtractionConfig.model_type, ExtractionConfig.modality, ExtractionConfig.mode, ExtractionConfig.layer))


def download_wmt(ds_name, lang_pair, save_path, cache_dir="./hfds_cache"):
    from datasets import load_dataset
    data = load_dataset(ds_name, lang_pair, cache_dir=cache_dir)
    data.save_to_disk(save_path)
    FileUtils.check_dirs(save_path)
    
    L1, L2 = lang_pair.split("-")
    for split in ['train', 'validation', 'test']:
        data_L1, data_L2 = [], []
        for ex in data[split]:
            data_L1.append(ex['translation'][L1])
            data_L2.append(ex['translation'][L2])
        FileUtils.save_file(data_L1, save_path + "/{}.{}".format(split, L1))
        FileUtils.save_file(data_L2, save_path + "/{}.{}".format(split, L2))


def wmt_to_txt(hfds_path, save_dir, src_lang="de", tgt_lang="en", subsets="train;validation;test"):
    from datasets import load_from_disk
    ds = load_from_disk(hfds_path)
    if not path.exists(save_dir):
        logging.warn("No directory found... Making by myself")
        os.mkdir(save_dir)
    for sub in subsets.split(";"):
        logging.info("Processing {}".format(sub))
        src_data, tgt_data = [], []
        for it in tqdm(ds[sub]):
            src_data.append(it['translation'][src_lang])
            tgt_data.append(it['translation'][tgt_lang])
        FileUtils.save_to_disk(src_data, save_dir+"/{}.{}".format(sub, src_lang))
        FileUtils.save_to_disk(tgt_data, save_dir+"/{}.{}".format(sub, tgt_lang))

    
def download_hf_model(model_name, save_dir):
    logging.info("Downloading {}...".format(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./hf_cache")
    model = AutoModel.from_pretrained(model_name, cache_dir="./hf_cache")
    is_mm = False if model.config.model_type not in MM_MODEL_TYPE_LIST else True
    if not path.exists(save_dir):
        os.makedirs(save_dir)
    if is_mm:
        processor = AutoProcessor.from_pretrained(model_name)
        processor.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    logging.info("Saved to {}...".format(save_dir))


def download_st_model(model_name, save_dir):
    from sentence_transformers import SentenceTransformer
    logging.info("Downloading {}...".format(model_name))
    model = SentenceTransformer(model_name)
    model.save(save_dir)
    logging.info("Saved to {}...".format(save_dir))

def extract_hidden_states(outputs, mask):
    if ExtractionConfig.mode == "cls":
        cls_pos = CLS_POS_DICT[ExtractionConfig.model_type]
        if not ExtractionConfig.is_mm:
            cat_repr = outputs.hidden_states[ExtractionConfig.layer][:, cls_pos]
        else:
            if ExtractionConfig.modality == "text":
                cat_repr = outputs.text_model_output.hidden_states[ExtractionConfig.layer][:, cls_pos['text']]
            elif ExtractionConfig.modality == "vision":
                cat_repr = outputs.vision_model_output.hidden_states[ExtractionConfig.layer][:, cls_pos['vision']]
            elif ExtractionConfig.modality == "all":
                cat_repr = torch.cat([
                    outputs.text_model_output.hidden_states[ExtractionConfig.layer][:, cls_pos['text']],
                    outputs.vision_model_output.hidden_states[ExtractionConfig.layer][:, cls_pos['vision']],
                    ], dim=-1)
            else:
                raise NotImplementedError
    elif ExtractionConfig.mode == "avg":
        if not ExtractionConfig.is_mm:
            cat_repr = HFUtils.avg_hidden_states(outputs.hidden_states[ExtractionConfig.layer], mask=mask)
        else:
            if ExtractionConfig.modality == "text":
                cat_repr = HFUtils.avg_hidden_states(outputs.text_model_output.hidden_states[ExtractionConfig.layer], mask=mask)
            elif ExtractionConfig.modality == "vision":
                cat_repr = HFUtils.avg_hidden_states(outputs.vision_model_output.hidden_states[ExtractionConfig.layer])
            elif ExtractionConfig.modality == "all":
                cat_repr = torch.cat([
                    HFUtils.avg_hidden_states(outputs.text_model_output.hidden_states[ExtractionConfig.layer], mask=mask),
                    HFUtils.avg_hidden_states(outputs.vision_model_output.hidden_states[ExtractionConfig.layer]),
                    ], dim=-1)
            else:
                raise NotImplementedError
    return cat_repr


def extract_repr_for_shard(model, tokenizer, shard_data, batch_size, device):
    s, shard_size = 0, len(shard_data)
    repr_data = []
    while s < shard_size:
        e = s + batch_size
        raw_batch = shard_data[s:e]
        if not ExtractionConfig.is_mm:
            batch = raw_batch
        else:
            batch = {
                "text": [it[0] for it in raw_batch],
                "images": [it[1] for it in raw_batch]
            }
        
        outputs, mask = HFUtils.encode(model, tokenizer, batch, device, is_mm=ExtractionConfig.is_mm)
        cat_repr = extract_hidden_states(outputs, mask=mask)
        cur_repr_data = cat_repr.cpu().unbind(dim=0)
        if ExtractionConfig.is_mm:
            cur_repr_data = list(zip(cur_repr_data, outputs.logits_per_image.diag().cpu().tolist()))
        repr_data += cur_repr_data
        s = e
    return repr_data


def extract_transformers_repr(model_path, input_file_pattern, saveprefix, save_step=1000000, batch_size=256, mode="cls_12", loaded_data=None, enable_fp16=False):
    if enable_fp16:
        model = AutoModel.from_pretrained(model_path).half().eval()
    else:
        model = AutoModel.from_pretrained(model_path).eval()
    ExtractionConfig.model_type = model.config.model_type
    if mode.startswith("cls") and ExtractionConfig.model_type not in CLS_POS_DICT:
        logging.error("The model type {} was not supported".format(ExtractionConfig.model_type))
        logging.error("Please choose model type in {}".format(CLS_POS_DICT.keys()))
        raise ValueError
    ExtractionConfig.is_mm = False if ExtractionConfig.model_type not in MM_MODEL_TYPE_LIST else True
    if ExtractionConfig.is_mm:
        tokenizer = AutoProcessor.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    logging.info("Finish to load {} model...".format(ExtractionConfig.model_type))
    mode_splits = mode.lower().split("_")
    if len(mode_splits) > 2:
        ExtractionConfig.mode, ExtractionConfig.layer, ExtractionConfig.modality = mode_splits
    else:
        ExtractionConfig.mode, ExtractionConfig.layer = mode_splits
        ExtractionConfig.modality = None
    ExtractionConfig.layer = int(ExtractionConfig.layer)
    if ExtractionConfig.modality is not None and ExtractionConfig.modality not in MODALITY_LIST:
        logging.error("Modality {} was not supported".format(ExtractionConfig.modality))
        logging.error("Please choose modality in {}".format(MODALITY_LIST))
        raise ValueError
    if ExtractionConfig.mode not in MODE_LIST:
        logging.error("Mode {} was not supported".format(ExtractionConfig.mode))
        logging.error("Please use `cls` mode")
        raise ValueError
    shard_sid, n_shard = 0, 0
    ExtractionConfig.show()
    if input_file_pattern is not None and loaded_data is None:
        data_iterator = FileUtils.data_iterator(input_file_pattern, shard_size=save_step)
    elif input_file_pattern is None and loaded_data is not None:
        data_iterator = []
        for i in range(0, len(loaded_data), save_step):
            data_iterator.append(loaded_data[i:i+save_step])
    for shard_data in data_iterator:
        shard_eid = shard_sid + len(shard_data)
        # if n_shard > 14:
        repr_data = extract_repr_for_shard(
            model, tokenizer, shard_data,
            batch_size=batch_size, device=device
        )
        repr_idx = list(range(shard_sid, shard_eid))
        assert len(repr_idx) == len(repr_data)
        if ExtractionConfig.is_mm:
            score_data = [it[1] for it in repr_data]
            repr_data = [it[0] for it in repr_data]
            FileUtils.save_to_disk(score_data, saveprefix+".{}.repr.score".format(n_shard))
        logging.info("Extractd ({}, {}) repr in shard {}".format(len(repr_data), repr_data[0].size(0), n_shard))
        # else:
        #     logging.info("Skip processed shard {}...".format(n_shard))
        FileUtils.save_to_disk(repr_data, saveprefix+".{}.repr.dat".format(n_shard))
        FileUtils.save_to_disk(repr_idx, saveprefix+".{}.repr.idx".format(n_shard))
        n_shard += 1
        shard_sid = shard_eid


def _extract_sentence_transformers_repr(proc_id, model_path, data, data_start_idx, saveprefix, batch_size=256, logging_step=1000000):
    model = SentenceTransformer(model_path).eval()
    device = torch.device("cuda:{}".format(proc_id)) if torch.cuda.device_count() > proc_id else torch.device("cpu")
    if device == torch.device("cpu"):
        logging.info("Cannot find available GPU device: cuda:{}".format(proc_id))
    model = model.to(device)
    batch_sents, batch_ids = [], []
    repr_data, repr_idx = [], []
    num_data = len(data)
    with torch.no_grad():
        for i, sent in enumerate(data):
            if (i+1)%logging_step == 0:
                logging.info("Proc-{} has processed {}/{} ({:.2f}%) sentences".format(proc_id, i+1, num_data, (i+1)/num_data * 100))
            if len(batch_sents) >= batch_size:
                batch_repr_data = model.encode(batch_sents, batch_size=batch_size, convert_to_tensor=True, device=device, show_progress_bar=False)
                repr_data.extend(batch_repr_data.cpu().unbind(dim=0))
                repr_idx.extend(batch_ids)
                batch_sents, batch_ids = [], []
            batch_sents.append(sent)
            batch_ids.append(i + data_start_idx)
    
    if batch_sents:
        batch_repr_data = model.encode(batch_sents, batch_size=batch_size, convert_to_tensor=True, device=device, show_progress_bar=False)
        repr_data.extend(batch_repr_data.cpu().unbind(dim=0))
        repr_idx.extend(batch_ids)

    FileUtils.save_to_disk(torch.stack(repr_data, dim=0), saveprefix+".{}.repr.dat".format(proc_id))
    FileUtils.save_to_disk(repr_idx, saveprefix+".{}.repr.idx".format(proc_id))


def extract_sentence_transformers_repr(model_path, input_file_path, saveprefix, batch_size=256, nproc=8, is_study_mode=False):
    data = FileUtils.load_file(input_file_path)
    if is_study_mode:
        data = data[:10000]
    data_shards = MPUtils.prepare_shards(data, nproc)
    args_list = []
    proc_data_start_id = 0
    
    for proc_id in range(nproc):
        args_list.append([proc_id, model_path, data_shards[proc_id], proc_data_start_id, saveprefix, batch_size])
        proc_data_start_id += len(data_shards[proc_id])

    MPUtils.mp_func(_extract_sentence_transformers_repr, args_list)

    
def main():
    fire.Fire({
        "download_hf_model": download_hf_model,
        "extract_transformers_repr": extract_transformers_repr,
        "extract_sentence_transformers_repr": extract_sentence_transformers_repr,
        "download_wmt": download_wmt,
        "wmt_to_txt": wmt_to_txt
    })

if __name__ == "__main__":
    main()
