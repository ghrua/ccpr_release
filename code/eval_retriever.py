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
from common_utils import override
import xml.etree.ElementTree as ET
import re
from glob import glob

def prepare_deen_annotation_data(data_dir="../align_data/DeEn", save_dir="../align_data/DeEn/"):
    src_detokenzier = DataUtils.get_moses_detokenizer("de")
    tgt_detokenzier = DataUtils.get_moses_detokenizer("en")
    src_data = FileUtils.load_file("{}/de.txt".format(data_dir), 'txt')
    tgt_data = FileUtils.load_file("{}/en.txt".format(data_dir), 'txt')
    align = FileUtils.load_file("{}/alignmentDeEn.talp".format(data_dir), 'txt')
    new_data, detok_src, detok_tgt = [], [], []
    for s, t, a in zip(src_data, tgt_data, align):
        if s.strip() and t.strip():
            new_a = []
            for it in a.split():
                if "-" in it:
                    u, v = it.split("-")
                else:
                    u, v = it.split("p")
                new_a.append("{}-{}".format(int(u)-1, int(v)-1))  
            new_data.append(" {##} ".join([s.lower(), t.lower(), " ".join(new_a)]))
            detok_src.append(src_detokenzier.detokenize(s.split()))
            detok_tgt.append(tgt_detokenzier.detokenize(t.split()))
    FileUtils.save_file(new_data, save_dir + "/aligned.test.deen", "txt")
    FileUtils.save_file(detok_src, save_dir + "/test.clean.de", "txt")
    FileUtils.save_file(detok_tgt, save_dir + "/test.clean.en", "txt")


def escape_xml_chars(xml_str):
    return xml_str.replace("&", "&amp;")


def prepare_czen_annotation_data(data_dir="../align_data/CzEn/merged_data/pcedt", save_dir="../align_data/CzEn/"):
    src_detokenzier = DataUtils.get_moses_detokenizer("cs")
    tgt_detokenzier = DataUtils.get_moses_detokenizer("en")
    new_data, detok_src, detok_tgt = [], [], []

    for fname in FileUtils.listdir(data_dir, return_full_name=True, only_file=True):
        if fname.endswith("wa"):
            with open(fname, 'r', encoding="utf-8") as file:
                logging.info("Loading from {}".format(fname))
                xml_data = escape_xml_chars(file.read())
                root = ET.fromstring(xml_data)
                for sentence in root.findall('s'):
                    sentence_id = sentence.get('id')
                    english_text = sentence.find('english').text
                    czech_text = sentence.find('czech').text
                    # this alignment is from en -> cs
                    sure_text = sentence.find('sure').text
                    possible_text = sentence.find('possible').text
                    new_a = []
                    for al in [sure_text, possible_text]:
                        if al is None or not al.strip():
                            continue
                        for it in al.split():
                            if "-" in it:
                                u, v = it.split("-")
                            else:
                                u, v = it.split("p")
                            new_a.append("{}-{}".format(int(v)-1, int(u)-1))  
                    new_data.append(" {##} ".join([czech_text.lower(), english_text.lower(), " ".join(new_a)]))
                    detok_src.append(src_detokenzier.detokenize(czech_text.split(), unescape=True))
                    detok_tgt.append(tgt_detokenzier.detokenize(english_text.split(), unescape=True))
        else:
            logging.info("Skip {}".format(fname))

    FileUtils.save_file(new_data, save_dir + "/aligned.test.csen", "txt")
    FileUtils.save_file(detok_src, save_dir + "/test.clean.cs", "txt")
    FileUtils.save_file(detok_tgt, save_dir + "/test.clean.en", "txt")


def prepare_roen_annotation_data(data_dir="../align_data/RoEn/Romanian-English.test", save_dir="../align_data/RoEn/"):
    
    def extract(text):
        pattern = r'<s snum=(\d+)> (.*?) </s>'
        match = re.search(pattern, text)
        snum_value, extracted_text = match.groups()
        return snum_value, extracted_text

    src_detokenzier = DataUtils.get_moses_detokenizer("ro")
    tgt_detokenzier = DataUtils.get_moses_detokenizer("en")
    new_data, detok_src, detok_tgt = [], [], []

    data_dict = {}
    for file_path in FileUtils.listdir(data_dir, return_full_name=True, only_file=True):
        logging.info("Found {}".format(file_path))
        file_type = FileUtils.check_file_type(file_path)
        if file_type == "r":
            for line in FileUtils.load_file(file_path, 'txt'):
                snum_value, extracted_text = extract(line)
                if snum_value in data_dict:
                    data_dict[snum_value]["ro"] = extracted_text
                else:
                    data_dict[snum_value] = {"ro": extracted_text, "en": "", "align": []} 
        elif file_type == "e":
            for line in FileUtils.load_file(file_path, 'txt'):
                snum_value, extracted_text = extract(line)
                if snum_value in data_dict:
                    data_dict[snum_value]["en"] = extracted_text        
                else:
                    data_dict[snum_value] = {"en": extracted_text, "ro": "", "align": []}
        elif file_type == "nonullalign":
            for line in FileUtils.load_file(file_path, 'txt'):
                snum_value, a, b, _ = line.split()
                if snum_value in data_dict:
                    data_dict[snum_value]["align"].append("{}-{}".format(a, b))
                else:
                    data_dict[snum_value] = {"en": "", "ro": "", "align": ["{}-{}".format(a, b)]}
        else:
            logging.info("Skip {}".format(file_path))
    
    for snum, val in data_dict.items():
        src, tgt, al = val['ro'], val['en'], val['align']
        new_a = []
        for it in al:
            if "-" in it:
                u, v = it.split("-")
            else:
                u, v = it.split("p")
            new_a.append("{}-{}".format(int(u)-1, int(v)-1))  
        if not new_a:
            continue
        new_data.append(" {##} ".join([src.lower(), tgt.lower(), " ".join(new_a)]))
        detok_src.append(src_detokenzier.detokenize(src.split(), unescape=True))
        detok_tgt.append(tgt_detokenzier.detokenize(tgt.split(), unescape=True))

    FileUtils.save_file(new_data, save_dir + "/aligned.test.roen", "txt")
    FileUtils.save_file(detok_src, save_dir + "/test.clean.ro", "txt")
    FileUtils.save_file(detok_tgt, save_dir + "/test.clean.en", "txt")


def collate_fn(batch_data, encode_tgt=True, pad_id=0):
    input_ids, phrase_batch_ids, phrase_start_ids, phrase_end_ids, phrase_text_list = [], [], [], [], []
    corpus_ids = []
    for bid, ex in enumerate(batch_data):
        input_ids.append(ex['ref_input_ids'] if encode_tgt else ex['src_input_ids'])
        cur_start_ids = ex['ref_phrase_start_ids'] if encode_tgt else ex['src_phrase_start_ids']
        cur_end_ids = ex['ref_phrase_end_ids'] if encode_tgt else ex['src_phrase_end_ids']
        phrase_batch_ids.extend([bid] * len(cur_start_ids))
        phrase_start_ids.extend(cur_start_ids)
        phrase_end_ids.extend(cur_end_ids)
        corpus_ids.extend([ex["corpus_id"]] * len(cur_start_ids))
        phrase_text_list.extend(ex['ref_phrase'] if encode_tgt else ex['src_phrase'])
    
    input_ids, attention_mask = TorchUtils.batchfy([torch.LongTensor(it) for it in input_ids], pad_id)
    phrase_batch_ids = torch.LongTensor(phrase_batch_ids)
    phrase_start_ids = torch.LongTensor(phrase_start_ids)
    phrase_end_ids = torch.LongTensor(phrase_end_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "phrase_batch_ids": phrase_batch_ids,
        "phrase_start_ids": phrase_start_ids,
        "phrase_end_ids": phrase_end_ids,
        "corpus_ids": corpus_ids,
        "phrase_text_list": phrase_text_list
    }


def encode_step(model, batch, model_type="ours_labse"):
    with torch.no_grad():
        if model_type.startswith("ours"):
            out = model.encoder_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            hidden_states = out.hidden_states[-1]
            phrase_batch_ids, phrase_start_ids, phrase_end_ids = batch['phrase_batch_ids'], batch['phrase_start_ids'], batch['phrase_end_ids']
            phrase_hidden_states_start = hidden_states[phrase_batch_ids, phrase_start_ids]
            phrase_hidden_states_end = hidden_states[phrase_batch_ids, phrase_end_ids]
            phrase_hidden_states = model.linear(torch.cat([phrase_hidden_states_start, phrase_hidden_states_end], dim=-1))
        else:
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            hidden_states = out.hidden_states[-1]
            phrase_batch_ids, phrase_start_ids, phrase_end_ids = batch['phrase_batch_ids'], batch['phrase_start_ids'], batch['phrase_end_ids']
            phrase_hidden_states_start = hidden_states[phrase_batch_ids, phrase_start_ids]
            phrase_hidden_states_end = hidden_states[phrase_batch_ids, phrase_end_ids]
            phrase_hidden_states = torch.cat([phrase_hidden_states_start, phrase_hidden_states_end], dim=-1)
        return phrase_hidden_states



def encode_dataset(model, ds, save_prefix, model_type="ours_labse", batch_size=256, indices=None, phrase_start_idx=0, encode_tgt=True, pad_id=0, device=torch.device("cuda")):
    if indices is None:
        indices = list(range(len(ds)))
    num_indices = len(indices)
    batch_data = []
    hidden_states_list, corpus_ids, phrase_text_list = [], [], []
    phrase_num = 0
    for i in indices:
        ex = ds[i]
        batch_data.append(ex)
        if len(batch_data) >= batch_size or (i+1) == num_indices:
            batch = collate_fn(batch_data, encode_tgt, pad_id)
            cur_corpus_ids = batch.pop("corpus_ids")
            phrase_num += len(cur_corpus_ids)
            cur_phrase_text_list = batch.pop("phrase_text_list")
            phrase_text_list.extend(cur_phrase_text_list)
            corpus_ids.extend(cur_corpus_ids)
            batch = TorchUtils.move_to_device(batch, device)
            phrase_hidden_states = encode_step(model, batch, model_type)

            hidden_states_list.append(phrase_hidden_states.cpu())
            batch_data = []
    
    FileUtils.save_file(torch.cat(hidden_states_list, dim=0), save_prefix + ".repr.dat")
    FileUtils.save_file(list(range(phrase_start_idx, phrase_start_idx + phrase_num)), save_prefix + ".repr.idx")
    FileUtils.save_file(corpus_ids, save_prefix + ".repr.sid")
    FileUtils.save_file(phrase_text_list, save_prefix + ".repr.txt")
    return phrase_start_idx + phrase_num


def encode_data(eval_config_path, **kwargs):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    args = FileUtils.load_file(eval_config_path)
    if kwargs:
        args = override(args, kwargs)
    FileUtils.check_dirs(args['save_dir'] )
    train_databin_prefix = args['train_databin_prefix']
    eval_databin_prefix = args['eval_databin_prefix']
    src_lang = args['eval_src_lang'].split(",")
    tgt_lang = args['eval_tgt_lang'].split(",")
    batch_size = args.get('batch_size', 256)
    model_type = args['model_type']

    if model_type == "labse" or model_type == "xlmr" or model_type == "mbert":
        model = AutoModel.from_pretrained(args["model_pretrained_dir"])
    elif model_type.startswith("ours"):
        model = get_model_class(args['model_name']).from_pretrained(args)
    else:
        raise NotImplementedError
    
    model = model.eval().to(device)

    if model_type.endswith("labse") or model_type.endswith("mbert"):
        pad_token_id = 0
    elif model_type.endswith("xlmr"):
        pad_token_id = 1
    else:
        raise NotImplementedError

    train_ds_args = args.copy()
    train_ds_args['data_pretrained_dir_prefix'] = train_databin_prefix
    train_ds_args['src_lang'] = args['train_src_lang']
    train_ds_args['tgt_lang'] = args['train_tgt_lang']
    train_ds = get_dataset_class(train_ds_args['ds_name'])(train_ds_args)
    num_train_ds = len(train_ds)
    logging.info("{} examples in the training datastore".format(num_train_ds))
    phrase_start_idx, src_shard_id, tgt_shard_id = 0, 0, 0
    phrase_start_idx = encode_dataset(model, train_ds, args['save_dir'] + "/data.tgt.{}".format(tgt_shard_id), model_type=args['model_type'], batch_size=batch_size, phrase_start_idx=phrase_start_idx, encode_tgt=True, pad_id=pad_token_id, device=device)
    tgt_shard_id += 1

    for sl, tl in zip(src_lang, tgt_lang):
        eval_ds_args = args.copy()
        eval_ds_args['data_pretrained_dir_prefix'] = eval_databin_prefix
        eval_ds_args['src_lang'] = sl
        eval_ds_args['tgt_lang'] = tl
        eval_ds = get_dataset_class(eval_ds_args['ds_name'])(eval_ds_args)
        num_eval_ds = len(eval_ds)
        logging.info("{} examples in the evaluation datastore".format(num_eval_ds))

        prev_phrase_start_idx = phrase_start_idx
        phrase_start_idx = encode_dataset(model, eval_ds, args['save_dir'] + "/data.tgt.{}".format(tgt_shard_id), model_type=args['model_type'], batch_size=batch_size, phrase_start_idx=phrase_start_idx, encode_tgt=True, pad_id=pad_token_id, device=device)

        src_phrase_start_idx = encode_dataset(model, eval_ds, args['save_dir'] + "/data.src.{}".format(sl), model_type=args['model_type'], batch_size=batch_size, phrase_start_idx=0, encode_tgt=False, pad_id=pad_token_id, device=device)

        golden_mapping = [(a, b) for a, b in zip(range(0, src_phrase_start_idx), range(prev_phrase_start_idx, phrase_start_idx))]
        FileUtils.save_file(golden_mapping,  args['save_dir'] + "/data.src.{}.golden.map".format(sl))
        src_shard_id += 1
        tgt_shard_id += 1


def retrieval_accuracy(retrieval_out_path, golden_mapping_path, topk=5, query_data_path=None, datastore_prefix=None, debug_file_path=None, human_indices_path=None):
    if datastore_prefix is not None:
        n_shrads = len(list(glob(datastore_prefix + "*.repr.txt")))
        phrase_text_list = FileUtils.load_shards(datastore_prefix, "repr.txt", n_shrads)
    else:
        phrase_text_list = []
    if query_data_path is not None:
        query_text_list = FileUtils.load_file(query_data_path)
    else:
        query_text_list = []
    golden_mapping = {a: b for a, b in FileUtils.load_file(golden_mapping_path)}
    retrieval_out = FileUtils.load_file(retrieval_out_path)
    query_ids, search_ids = retrieval_out['query_ids'], retrieval_out['search_ids']
    n_query = len(query_ids)
    matched = 0
    debug_data = []
    if human_indices_path is not None:
        selected_indices = FileUtils.load_file(human_indices_path)
        n_query = len(selected_indices)
    else:
        selected_indices = list(range(n_query))
    # for ptr in range(n_query):
    for ptr in selected_indices:
        qid = query_ids[ptr]
        candidates = [int(si) for si in search_ids[ptr][:topk]]
        if phrase_text_list and query_text_list:
            candidates_text = [phrase_text_list[si] for si in candidates]
            query_text = query_text_list[qid]
            debug_data.append({"query": query_text, "retrieval": candidates_text, "golden": phrase_text_list[golden_mapping[qid]]})
        if golden_mapping[qid] in candidates:
            matched += 1
    if debug_file_path is not None:
        FileUtils.save_file(debug_data, debug_file_path, "json")
    logging.info("Acc@{}:\t{:.3f}".format(topk, matched / n_query * 100))


if __name__ == "__main__":
    fire.Fire({
        "prepare_deen_annotation_data": prepare_deen_annotation_data,
        "prepare_czen_annotation_data": prepare_czen_annotation_data,
        "prepare_roen_annotation_data": prepare_roen_annotation_data,
        "retrieval_accuracy": retrieval_accuracy,
        "encode_data": encode_data
    })