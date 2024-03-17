import sentencepiece as spm
import numpy as np
import fire
import json
import os
import os.path as path
import logging
from tqdm import tqdm
import torch
from collections import Counter
from mytools.tool_utils import FileUtils, DataUtils, HFUtils, FileType, LANGUAGE_LIST
import random
from transformers import AutoModel, AutoTokenizer
from mytools.tool_utils import FileUtils
from glob import glob
import time
random.seed(10086)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)



def test_retrieval_after_addvalid(valid_retrieval_prefix, valid_data_path, retriever_tokenizer_path):
    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_tokenizer_path)
    tokenized_data = []
    for line in FileUtils.load_file(valid_data_path):
        tokens = retriever_tokenizer(line)['input_ids']
        tokens = retriever_tokenizer.convert_ids_to_tokens(tokens)
        tokenized_data.append(" ".join(tokens))
    total_match, n_sent, phrase_num = 0, 0, 0
    for valid_json_file in glob(valid_retrieval_prefix + "*.json"):
        json_data = FileUtils.load_file(valid_json_file)
        for it in json_data:
            if not it:
                continue
            try:
                sent_id = int(it[0].split("\t")[1])
            except Exception:
                print(sent_id)
            for s in it[1:]:
                if s in tokenized_data[sent_id]:
                    total_match += 1
                phrase_num += 1
            n_sent += 1
    print(phrase_num / n_sent)
    print(total_match / n_sent)


def test_tokenizer_time(tgt_corpus_data_path, retriever_tokenizer_path):
    corpus_data = FileUtils.load_file(tgt_corpus_data_path)
    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_tokenizer_path)
    import time
    logging.info("Start simple tokenization...")
    start_time = time.time()
    tokenized_data = []
    for line in corpus_data[:1000]:
        tokenized_data.append(retriever_tokenizer(line)['input_ids'])
    end_time = time.time()
    logging.info("End simple tokenization... Cost of time: {}".format(end_time - start_time))
    
    logging.info("Start converting to ids tokenization...")
    start_time = time.time()
    for line in tokenized_data:
        _ = retriever_tokenizer.convert_ids_to_tokens(line)
    end_time = time.time()
    logging.info("End converting to ids tokenization... Cost of time: {}".format(end_time - start_time))

    logging.info("Start decoding...")
    start_time = time.time()
    for line in tokenized_data:
        _ = retriever_tokenizer.decode(line)
    end_time = time.time()
    logging.info("End decoding... Cost of time: {}".format(end_time - start_time))


def tokenize_corpus(tgt_corpus_data_path, retriever_tokenizer_path, save_tag="xlmr", batch_size=256):
    corpus_data = FileUtils.load_file(tgt_corpus_data_path)
    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_tokenizer_path)
    corpus_ids, corpus_tokens, batch = [], [], []
    n_sent = len(corpus_data)
    for i in tqdm(range(n_sent)):
        batch.append(corpus_data[i])
        if len(batch) >= batch_size:
            batch_out = retriever_tokenizer(batch, truncation=True)
            corpus_ids.extend(batch_out['input_ids'])
            corpus_tokens.extend([retriever_tokenizer.convert_ids_to_tokens(it) for it in batch_out['input_ids']])
            batch = []
        
    if batch:
        batch_out = retriever_tokenizer(batch, truncation=True)
        corpus_ids.extend(batch_out['input_ids'])
        corpus_tokens.extend([retriever_tokenizer.convert_ids_to_tokens(it) for it in batch_out['input_ids']])
    logging.info("{}\t{}\t{}".format(len(corpus_data), len(corpus_ids), len(corpus_tokens)))
    FileUtils.save_file(corpus_ids, tgt_corpus_data_path + ".{}.tok.ids".format(save_tag))
    FileUtils.save_file(corpus_tokens, tgt_corpus_data_path + ".{}.tok.str".format(save_tag))


def decode_xlmr_with_brackets(input_tokens, insert_ids, sepcial_token=[]):
    s, e = insert_ids
    new_input_tokens = [tk for tk in input_tokens]
    new_input_tokens.insert(e+1, "]]")
    new_input_tokens[s] = "▁[[" + new_input_tokens[s] if not new_input_tokens[s].startswith("▁") else "▁[[" + new_input_tokens[s][1:]
    text = ""
    for tk in new_input_tokens:
        if tk in sepcial_token:
            continue
        if tk.startswith("▁"):
            text += " " + tk[1:]
        else:
            text += tk
    return text.strip()


def decode_labse_with_brackets(input_tokens, insert_ids, sepcial_token=None):
    s, e = insert_ids
    new_input_tokens = [tk for tk in input_tokens]
    new_input_tokens.insert(e+1, "##]]")
    new_input_tokens[s] = "[[" + new_input_tokens[s] if not new_input_tokens[s].startswith("##") else "##[[" + new_input_tokens[s][2:]
    text = ""
    for tk in new_input_tokens:
        if tk in sepcial_token:
            continue
        if tk.startswith("##"):
            text += tk[2:]
        else:
            text += " " + tk
    return text.strip()


def merge_analysis_data(analysis_data_paths, save_path, topk=-1):
    if isinstance(analysis_data_paths, str):
        analysis_data_paths = analysis_data_paths.split(",")
    
    data_list = []
    for data_path in analysis_data_paths:
        phrase_data = FileUtils.load_file(data_path, "json")
        phrase_data = {
            it['id']: it for it in phrase_data
        }
        data_list.append(phrase_data)
    main_data = data_list[0]

    aux_data_list = data_list[1:]

    key_gen_func = lambda x: x if isinstance(x, str) else tuple(x)
    for idx, ex in main_data.items():
        main_alignment = {key_gen_func(it[0]): it[1:] for it in ex['alignment']}
        for aux_data in aux_data_list:
            if idx in aux_data:
                aux_alignment = {key_gen_func(it[0]): it[1:] for it in aux_data[idx]['alignment']}
                for p, al_p_list in aux_alignment.items():
                    if p in main_alignment:
                        main_alignment[p].extend(al_p_list)
                    else:
                        main_alignment[p] = al_p_list
        if topk > 0:
            main_data[idx]['alignment'] = [(p, *sorted(al_p_list, key=lambda x: -x[-1])[:topk]) for p, al_p_list in main_alignment.items()]
        else:
            main_data[idx]['alignment'] = [(p, *sorted(al_p_list, key=lambda x: -x[-1])) for p, al_p_list in main_alignment.items()]
    main_data = list(main_data.values())
    FileUtils.save_file(main_data, save_path, "json")


def assemble_sentence_retrieval_data(datastore_corpus_data_path, src_data_path, ref_data_path, retrieval_out_path, save_path, topk=10):
    corpus = FileUtils.load_file(datastore_corpus_data_path)
    src_data = FileUtils.load_file(src_data_path)
    ref_data = FileUtils.load_file(ref_data_path)
    retrieval_out = FileUtils.load_file(retrieval_out_path)
    query_ids, search_ids, search_dist = retrieval_out['query_ids'], retrieval_out['search_ids'], retrieval_out['search_dist']
    output_data = []
    for i in range(len(query_ids)):
        sid = int(query_ids[i])
        output_data.append({
            "id": sid,
            "src": src_data[i],
            "ref": ref_data[i],
            "retrieved_sentences": [corpus[j] for j in search_ids[i][:topk]],
            "dist": [float(d) for d in search_dist[i][:topk]]
        })
    FileUtils.save_file(output_data, save_path, "json")


def assemble_analysis_data(src_data_prefix, datastore_data_prefix, src_corpus_data_path, ref_corpus_data_path, datastore_corpus_data_path, retriever_tokenizer_path, retrieval_tag="", topk=1, use_raw_phrase=False, model_type="xlmr", chunk_size=6, chunk_size_epsilon=2, avoid_same_sentid=False, tgt_lang="en"):
    n_src_shrads, n_tgt_shards = len(list(glob(src_data_prefix + "*.repr.sid"))), len(list(glob(datastore_data_prefix + "*.repr.dat")))
    logging.info("{} src shards and {} tgt shards".format(n_src_shrads, n_tgt_shards))
    src_shard_ids = list(range(n_src_shrads))
    logging.info("Running on all shards: {}".format(src_shard_ids))
    src_phrase_sentid_list = FileUtils.load_shards(src_data_prefix, "repr.sid", n_src_shrads)
    src_phrase_spos_list = FileUtils.load_shards(src_data_prefix, "repr.spos", n_src_shrads)
    src_phrase_epos_list = FileUtils.load_shards(src_data_prefix, "repr.epos", n_src_shrads)
    src_phrase_text_list = FileUtils.load_shards(src_data_prefix, "repr.txt", n_src_shrads)

    tgt_phrase_text_list = FileUtils.load_shards(datastore_data_prefix, "repr.txt", n_tgt_shards)
    tgt_phrase_sentid_list = FileUtils.load_shards(datastore_data_prefix, "repr.sid", n_tgt_shards)
    tgt_phrase_start_pos_list = FileUtils.load_shards(datastore_data_prefix, "repr.spos", n_tgt_shards)
    tgt_phrase_end_pos_list = FileUtils.load_shards(datastore_data_prefix, "repr.epos", n_tgt_shards)

    src_corpus = FileUtils.load_file(src_corpus_data_path)
    ref_corpus = FileUtils.load_file(ref_corpus_data_path)
    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_tokenizer_path)

    # datastore_corpus = FileUtils.load_file(datastore_corpus_data_path)
    corpus_toks = FileUtils.load_file(datastore_corpus_data_path + ".{}.tok.str".format(model_type))

    moses_tokenizer = DataUtils.get_moses_detokenizer(tgt_lang)

    def is_complete_token(tok):
        if model_type == "xlmr":
            return tok.startswith("▁")
        elif model_type == "labse":
            return not tok.startswith("##")
        else:
            raise NotImplementedError

    def find_feasible_chunk(tokens, start_id, end_id):
        n_toks = len(tokens)
        left_chunk_size = chunk_size - (end_id - start_id)
        new_start_id = max(0, start_id - left_chunk_size // 2)
        new_end_id = min(n_toks, end_id + 1 + left_chunk_size // 2)
        ptr = new_start_id
        while not is_complete_token(tokens[ptr]) and ptr > 0 and (new_start_id - ptr) <= chunk_size_epsilon:
            ptr -= 1
        new_start_id = ptr
        ptr = new_end_id
        while not is_complete_token(tokens[ptr-1]) and ptr < n_toks and (ptr - new_end_id) <= chunk_size_epsilon:
            ptr += 1
        new_end_id = ptr
        return tokens[new_start_id:new_end_id], start_id - new_start_id, end_id - new_start_id

    prev_sentid = 0
    output_data, tmp_alignment = [], []
    for src_shard_id in src_shard_ids:
        if retrieval_tag:
            shard = FileUtils.load_file("{}.{}.{}.{}".format(src_data_prefix, src_shard_id, retrieval_tag, "retrieval.out.pt"))
        else:
            shard = FileUtils.load_file("{}.{}.{}".format(src_data_prefix, src_shard_id, "retrieval.out.pt"))
        query_ids, search_ids, search_dist = shard['query_ids'], shard['search_ids'], shard['search_dist']
        
        item_dict = {}
        tmp_text_list = set()
        for ptr in range(len(query_ids)):
            qid = query_ids[ptr]
            sid = src_phrase_sentid_list[qid]
            spos = src_phrase_spos_list[qid]
            epos = src_phrase_epos_list[qid]
            align_data = []
            if sid != prev_sentid:
                item_dict['id'] = prev_sentid
                item_dict['src_sent'] = src_corpus[prev_sentid]
                item_dict['tgt_sent'] = ref_corpus[prev_sentid]
                item_dict['alignment'] = tmp_alignment
                output_data.append(item_dict)
                prev_sentid = sid
                tmp_text_list = set()
                item_dict, tmp_alignment = {}, []
            if use_raw_phrase:
                src_phrase = " ".join(src_phrase_text_list[qid].split("▁")).strip()
            else:
                src_phrase = retriever_tokenizer.decode(retriever_tokenizer.convert_tokens_to_ids(src_phrase_text_list[qid].split()))
            align_data.append((src_phrase, spos, epos))
            unique_tgt_phrase_ids = []
            for phrase_idx, phrase_dist in zip(search_ids[ptr], search_dist[ptr]):

                phrase_sid = tgt_phrase_sentid_list[phrase_idx]
                if avoid_same_sentid and phrase_sid == sid:
                    continue
                tgt_sent_tokens = corpus_toks[phrase_sid]
                start_id, end_id = tgt_phrase_start_pos_list[phrase_idx], tgt_phrase_end_pos_list[phrase_idx]
                chunk_tokens, new_start_id, new_end_id = find_feasible_chunk(tgt_sent_tokens, start_id, end_id)
                if model_type == "xlmr":
                    tgt_sent_with_brackets = decode_xlmr_with_brackets(tgt_sent_tokens, (start_id, end_id), sepcial_token=retriever_tokenizer.special_tokens_map.values())
                elif model_type == "labse":
                    tgt_sent_with_brackets = decode_labse_with_brackets(tgt_sent_tokens, (start_id, end_id), sepcial_token=retriever_tokenizer.special_tokens_map.values())
                    tgt_sent_with_brackets = moses_tokenizer.detokenize(tgt_sent_with_brackets.split())
                else:
                    tgt_sent_with_brackets = ""

                decoded_chunk = retriever_tokenizer.decode(retriever_tokenizer.convert_tokens_to_ids(chunk_tokens), skip_special_tokens=True)
                tgt_phrase = tgt_phrase_text_list[phrase_idx]

                if use_raw_phrase:
                    tgt_phrase = " ".join(tgt_phrase_text_list[phrase_idx].split("▁")).strip()
                else:
                    tgt_phrase = retriever_tokenizer.decode(retriever_tokenizer.convert_tokens_to_ids(tgt_phrase_text_list[phrase_idx].split()))
                if tgt_phrase in tmp_text_list:
                    continue
                # align_data.append((phrase_sid, tgt_phrase, decoded_chunk, datastore_corpus[phrase_sid], phrase_dist))
                align_data.append((phrase_sid, tgt_phrase, decoded_chunk, tgt_sent_with_brackets, phrase_dist))
                tmp_text_list.add(tgt_phrase)
                unique_tgt_phrase_ids.append(phrase_idx)
                if len(unique_tgt_phrase_ids) >= topk:
                     break
            tmp_alignment.append(align_data)

        if tmp_alignment:
            item_dict['id'] = prev_sentid
            item_dict['src_sent'] = src_corpus[prev_sentid]
            item_dict['tgt_sent'] = ref_corpus[prev_sentid]
            item_dict['alignment'] = tmp_alignment
            output_data.append(item_dict)

    FileUtils.save_file(output_data, src_data_prefix + ".analysis.top{}.json".format(topk), "json")
    


def assemble_chunk_data(src_data_prefix, tgt_data_prefix, tgt_corpus_data_prefix, retriever_tokenizer_path, chunk_size=13, chunk_size_epsilon=4, topk=3, logging_step=320000, run_shards="", min_dist=0, model_type="xlmr", retrieval_tag=""):
    n_src_shrads, n_tgt_shards = len(list(glob(src_data_prefix + "*.repr.sid"))), len(list(glob(tgt_data_prefix + "*.repr.dat")))
    logging.info("{} src shards and {} tgt shards".format(n_src_shrads, n_tgt_shards))
    if run_shards:
        src_shard_ids = run_shards
        logging.info("Running on selected shards: {}".format(src_shard_ids))
    else:
        src_shard_ids = list(range(n_src_shrads))
        logging.info("Running on all shards: {}".format(src_shard_ids))
    # retrieval_out_data = FileUtils.load_shards(src_data_prefix, "retrieval.out.pt", n_src_shrads)
    src_phrase_sentid_list = FileUtils.load_shards(src_data_prefix, "repr.sid", n_src_shrads)
    src_phrase_text_list = FileUtils.load_shards(src_data_prefix, "repr.txt", n_src_shrads)

    tgt_phrase_text_list = FileUtils.load_shards(tgt_data_prefix, "repr.txt", n_tgt_shards)
    tgt_phrase_sentid_list = FileUtils.load_shards(tgt_data_prefix, "repr.sid", n_tgt_shards)
    tgt_phrase_start_pos_list = FileUtils.load_shards(tgt_data_prefix, "repr.spos", n_tgt_shards)
    tgt_phrase_end_pos_list = FileUtils.load_shards(tgt_data_prefix, "repr.epos", n_tgt_shards)

    chunk_list, chunk_input_ids, spos_list, epos_list, sid_list = [], [], [], [], []
    stats = dict()

    corpus_ids = FileUtils.load_file(tgt_corpus_data_prefix + ".ids")
    corpus_toks = FileUtils.load_file(tgt_corpus_data_prefix + ".str")
    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_tokenizer_path)
    # mt_tokenizer = FileUtils.load_spm(mt_tokenizer_path)

    prev_sentid = -1
    step = 0

    def is_complete_token(tok):
        if model_type == "xlmr":
            return tok.startswith("▁")
        elif model_type == "bert":
            return not tok.startswith("##")
        else:
            raise NotImplementedError
    
    def find_feasible_chunk(tokens, start_id, end_id):
        n_toks = len(tokens)
        left_chunk_size = chunk_size - (end_id - start_id)
        new_start_id = max(0, start_id - left_chunk_size // 2)
        new_end_id = min(n_toks, end_id + 1 + left_chunk_size // 2)
        ptr = new_start_id
        while not is_complete_token(tokens[ptr]) and ptr > 0 and (new_start_id - ptr) <= chunk_size_epsilon:
            ptr -= 1
        new_start_id = ptr
        ptr = new_end_id
        while not is_complete_token(tokens[ptr-1]) and ptr < n_toks and (ptr - new_end_id) <= chunk_size_epsilon:
            ptr += 1
        new_end_id = ptr
        return tokens[new_start_id:new_end_id], start_id - new_start_id, end_id - new_start_id
    
    for src_shard_id in src_shard_ids:
        if retrieval_tag:
            shard = FileUtils.load_file("{}.{}.{}.{}".format(src_data_prefix, src_shard_id, retrieval_tag, "retrieval.out.pt"))
        else:
            shard = FileUtils.load_file("{}.{}.{}".format(src_data_prefix, src_shard_id, "retrieval.out.pt"))
        query_ids, search_ids, search_dist = shard['query_ids'], shard['search_ids'], shard['search_dist']
        tmp_chunk_list, tmp_chunk_input_ids, tmp_spos_list, tmp_epos_list = [], [], [], []
        tmp_text_list = set()
        print(len(query_ids))
        for ptr in range(len(query_ids)):
            if (step + 1) % logging_step == 0:
                logging.info("Processed {} phrases".format(step+1))
            step += 1
            qid = query_ids[ptr]
            sid = src_phrase_sentid_list[qid]
            if sid != prev_sentid:
                if tmp_chunk_list:
                    sid_list.append(prev_sentid)
                    if len(tmp_chunk_list) > stats.get("max_len_retrival", 0):
                        stats['max_len_retrival'] = len(tmp_chunk_list)
                    tmp_chunk_list = ["Sent ID:\t{}".format(prev_sentid)] + tmp_chunk_list
                    chunk_list.append(tmp_chunk_list)
                    chunk_input_ids.append(tmp_chunk_input_ids)
                    assert len(tmp_chunk_input_ids) == len(tmp_spos_list) == len(tmp_epos_list)
                    spos_list.append(tmp_spos_list)
                    epos_list.append(tmp_epos_list)
                prev_sentid = sid
                tmp_chunk_list, tmp_chunk_input_ids, tmp_spos_list, tmp_epos_list = [], [], [], []
                tmp_text_list = set()

            unique_tgt_phrase_ids = []
            for phrase_idx, phrase_dist in zip(search_ids[ptr], search_dist[ptr]):
                phrase_sid = tgt_phrase_sentid_list[phrase_idx]
                retriever_tokenizer_out = corpus_ids[phrase_sid]
                tgt_sent_tokens = corpus_toks[phrase_sid]
                start_id, end_id = tgt_phrase_start_pos_list[phrase_idx], tgt_phrase_end_pos_list[phrase_idx]
                chunk_tokens, new_start_id, new_end_id = find_feasible_chunk(tgt_sent_tokens, start_id, end_id)

                decoded_chunk = " ".join(chunk_tokens)
                tgt_phrase = tgt_phrase_text_list[phrase_idx]
                chunk_token_ids = retriever_tokenizer.convert_tokens_to_ids(chunk_tokens)
                
                special_start, special_end = retriever_tokenizer_out[0], retriever_tokenizer_out[-1]
                if chunk_token_ids[0] != special_start:
                    chunk_token_ids = [special_start] + chunk_token_ids
                    new_start_id += 1
                    new_end_id += 1
                if chunk_token_ids[-1] != special_end:
                    chunk_token_ids = chunk_token_ids + [special_end]
                chunk_token_ids = np.array(chunk_token_ids)
                # retriever_tokenizer_out, retriever_offset = retriever_tokenizer(tgt_sent, return_offsets_mapping=True, truncation=True), retriever_tokenizer_out['offset_mapping']
                # tgt_phrase = tgt_phrase_text_list[phrase_idx]
                # start_id, end_id = tgt_phrase_start_pos_list[phrase_idx], tgt_phrase_end_pos_list[phrase_idx]
                # mt_tokens, mt_offset = DataUtils.sp_encode(mt_tokenizer, tgt_sent, return_offsets_mapping=True)
                # offset_map = DataUtils.map_tokenized_sents(retriever_offset, mt_offset)
                # if not offset_map[start_id] or not offset_map[end_id]:
                #     continue
                # mt_chunk, new_start_id, new_end_id = find_feasible_chunk(mt_tokens, offset_map[start_id], offset_map[end_id])
                # mt_input_ids = [mt_tokenizer.piece_to_id(tok) for tok in mt_chunk]
                # decoded_mt_chunk = mt_tokenizer.decode(mt_input_ids)
                if phrase_dist <= min_dist or tgt_phrase in tmp_text_list or decoded_chunk in tmp_text_list or phrase_sid == sid:
                    continue
                tmp_chunk_list.append(decoded_chunk)
                tmp_chunk_input_ids.append(chunk_token_ids)
                unique_tgt_phrase_ids.append(phrase_idx)
                tmp_spos_list.append(new_start_id)
                tmp_epos_list.append(new_end_id)
                tmp_text_list.add(tgt_phrase)
                tmp_text_list.add(decoded_chunk)
                if len(unique_tgt_phrase_ids) >= topk:
                     break
            if not unique_tgt_phrase_ids:
                continue

        if tmp_chunk_input_ids:
            sid_list.append(prev_sentid)
            chunk_list.append(tmp_chunk_list)
            chunk_input_ids.append(tmp_chunk_input_ids)
            spos_list.append(tmp_spos_list)
            epos_list.append(tmp_epos_list)
        if retrieval_tag:
            FileUtils.save_file(sid_list, src_data_prefix + ".{}.{}".format(retrieval_tag, src_shard_id) + ".pbnmt.chunk.sid")
            FileUtils.save_file(chunk_list, src_data_prefix + ".{}.{}".format(retrieval_tag, src_shard_id) + ".pbnmt.chunk.json", file_type="json")
            FileUtils.save_file(chunk_input_ids, src_data_prefix  + ".{}.{}".format(retrieval_tag, src_shard_id) + ".pbnmt.chunk.ids")
            FileUtils.save_file(spos_list, src_data_prefix + ".{}.{}".format(retrieval_tag, src_shard_id) + ".pbnmt.chunk.spos")
            FileUtils.save_file(epos_list, src_data_prefix + ".{}.{}".format(retrieval_tag, src_shard_id) + ".pbnmt.chunk.epos")
        else:
            FileUtils.save_file(sid_list, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.chunk.sid")
            FileUtils.save_file(chunk_list, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.chunk.json", file_type="json")
            FileUtils.save_file(chunk_input_ids, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.chunk.ids")
            FileUtils.save_file(spos_list, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.chunk.spos")
            FileUtils.save_file(epos_list, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.chunk.epos")
        chunk_list, chunk_input_ids, spos_list, epos_list, sid_list = [], [], [], [], []

    print("------------------ Stats ------------------")
    for k, v in stats.items():
        print("{}: {}".format(k, v))

def mix_pbnmt_data(main_data_prefix, aux_data_prefix, mixed_data_save_prefix, replace_ratio=0.2):
    def load_pbnmt_resources(phrase_data_prefix):
        data = dict()
        if FileUtils.exists("{}.spos".format(phrase_data_prefix)):
            data['phrase_start_pos'] = FileUtils.load_file("{}.spos".format(phrase_data_prefix))
            data['phrase_end_pos'] = FileUtils.load_file("{}.epos".format(phrase_data_prefix))
            data['phrase_sentid_to_realid'] = FileUtils.load_file("{}.sid".format(phrase_data_prefix))
            data['phrase_ids'] = FileUtils.load_file("{}.rid".format(phrase_data_prefix))
            data['phrase_hidden_states'] = torch.from_numpy(FileUtils.load_file("{}.npy".format(phrase_data_prefix)))
        else:
            raise ValueError("{}.* don't exist".format(phrase_data_prefix))
        return data
    
    main_data = load_pbnmt_resources(main_data_prefix)
    aux_data = load_pbnmt_resources(aux_data_prefix)
    aux_data['phrase_sentid_to_realid'] = {j: i for i, j in enumerate(aux_data['phrase_sentid_to_realid'])}
    n_miss_sent, n_empty_repr = 0, 0
    for realid, sentid in enumerate(main_data['phrase_sentid_to_realid']):
        if sentid not in aux_data['phrase_sentid_to_realid']:
            n_miss_sent += 1
            continue
        else:
            main_h_si = 0 if realid == 0 else main_data['phrase_ids'][realid-1]
            main_h_ei = main_data['phrase_ids'][realid]
            
            aux_realid = aux_data['phrase_sentid_to_realid'][sentid]
            aux_h_si = 0 if aux_realid == 0 else aux_data['phrase_ids'][aux_realid-1]
            aux_h_ei = aux_data['phrase_ids'][aux_realid]

            replaced_num = max(int(replace_ratio * (main_h_ei - main_h_si)), 0)
            replaced_num = min(replaced_num, aux_h_ei - aux_h_si)
            if replaced_num == 0:
                n_empty_repr += 1
                continue
            replaced_main_indices = np.array(random.sample(list(range(main_h_si, main_h_ei)), k=replaced_num))
            selected_aux_indices = np.array(random.sample(list(range(aux_h_si, aux_h_ei)), k=replaced_num))
            main_data['phrase_hidden_states'][replaced_main_indices] = aux_data['phrase_hidden_states'][selected_aux_indices]
            for enum_i, rep_id in enumerate(replaced_main_indices):
                sel_rep_id = selected_aux_indices[enum_i]
                assert 0 <= sel_rep_id - aux_h_si < len(aux_data['phrase_start_pos'][aux_realid])
                assert 0 <= rep_id - main_h_si < len(main_data['phrase_start_pos'][realid])
                main_data['phrase_start_pos'][realid][rep_id - main_h_si] = aux_data['phrase_start_pos'][aux_realid][sel_rep_id - aux_h_si]
                main_data['phrase_end_pos'][realid][rep_id - main_h_si] = aux_data['phrase_end_pos'][aux_realid][sel_rep_id - aux_h_si]
    
    logging.info("{} sents are missed".format(n_miss_sent))
    logging.info("{} sents are empty".format(n_empty_repr))

    FileUtils.save_file(main_data['phrase_start_pos'], "{}.spos".format(mixed_data_save_prefix))
    FileUtils.save_file(main_data['phrase_end_pos'], "{}.epos".format(mixed_data_save_prefix))
    FileUtils.save_file(main_data['phrase_sentid_to_realid'], "{}.sid".format(mixed_data_save_prefix))
    FileUtils.save_file(main_data['phrase_ids'], "{}.rid".format(mixed_data_save_prefix))
    FileUtils.save_file(main_data['phrase_hidden_states'], "{}.npy".format(mixed_data_save_prefix))



def analyze_retrieval_dist(src_data_prefix, tgt_data_prefix, topk=10, is_train_data=False):
    n_src_shrads, n_tgt_shards = len(list(glob(src_data_prefix + "*.repr.sid"))), len(list(glob(tgt_data_prefix + "*.repr.sid")))
    src_shard_ids = list(range(n_src_shrads))
    logging.info("{} src shards".format(n_src_shrads))
    src_phrase_sentid_list = FileUtils.load_shards(src_data_prefix, "repr.sid", n_src_shrads)
    tgt_phrase_text_list = FileUtils.load_shards(tgt_data_prefix, "repr.txt", n_tgt_shards)
    tgt_phrase_sentid_list = FileUtils.load_shards(tgt_data_prefix, "repr.sid", n_tgt_shards)

    prev_sentid = -1
    cur_dist, dists, tmp_text_list = [], [], set()
    for src_shard_id in src_shard_ids:
        shard = FileUtils.load_file("{}.{}.{}".format(src_data_prefix, src_shard_id, "retrieval.out.pt"))
        query_ids, search_ids, search_dist = shard['query_ids'], shard['search_ids'], shard['search_dist']
        for ptr in range(len(query_ids)):
            qid = query_ids[ptr]
            sid = src_phrase_sentid_list[qid]
            if sid != prev_sentid:
                prev_sentid = sid
                if cur_dist:
                    dists.append(cur_dist)
                    cur_dist, tmp_text_list = [], set()
            for phrase_idx, phrase_dist in zip(search_ids[ptr], search_dist[ptr]):
                phrase_text = tgt_phrase_text_list[phrase_idx]
                phrase_sid = tgt_phrase_sentid_list[phrase_idx]
                if phrase_text in tmp_text_list or (is_train_data and phrase_sid == sid) or len(cur_dist) >= topk:
                    continue
                tmp_text_list.add(phrase_text)
                cur_dist.append(phrase_dist)
    
    # analysis 1: overall dist
    def overall_avg(dists):
        sum_dist, n_dist = 0, 0
        for it in dists:
            for d in it:
                sum_dist += d
                n_dist += 1
        return sum_dist / n_dist
    
    # analysis 1: bin dist
    def stats_bins(dists):
        bin_size, max_size = 5, 100
        nbin = (max_size + bin_size - 1) // bin_size
        orders = [i * bin_size for i in list(range(nbin))[::-1]]
        bins = {i: 0 for i in orders}
        total_n = sum([len(it) for it in dists])
        for it in dists:
            for d in it:
                for i in orders:
                    if d > i:
                        bins[i] += 1/total_n * 100
                        break
        return bins
    print(overall_avg(dists))
    print(stats_bins(dists))


def remap_phrase(from_data_prefix, to_data_prefix, max_step=200):
    n_from_shrads, n_to_shards = len(list(glob(from_data_prefix + "*.repr.txt"))), len(list(glob(to_data_prefix + "*.repr.txt")))
    from_phrase_text_list = FileUtils.load_shards(from_data_prefix, "repr.txt", n_from_shrads)
    to_phrase_text_list = FileUtils.load_shards(to_data_prefix, "repr.txt", n_to_shards)
    i, j, max_i, max_j = 0, 0, len(from_phrase_text_list), len(to_phrase_text_list)
    map_dict = dict()

    def check_match_level(a, b):
        n = 0
        while from_phrase_text_list[a] == to_phrase_text_list[b]:
            n += 1
            if n >= 100:
                break
            a += 1
            b += 1
        return n

    while i < max_i and j < max_j:
        if from_phrase_text_list[i] == to_phrase_text_list[j]:
            map_dict[i] = j
            i += 1
            j += 1
        else:
            temp_i, temp_j = i, j
            for step in range(max_step):
                if from_phrase_text_list[i+step] == to_phrase_text_list[j]:
                    if check_match_level(i+step, j) < 5:
                        continue
                    else:
                        i = i+step
                        break
                    
            for step in range(max_step):
                if from_phrase_text_list[i] == to_phrase_text_list[j+step]:
                    if check_match_level(i, j+step) < 5:
                        continue
                    else:
                        j = j+step
                        break
            if i == temp_i and j == temp_j:
                import pdb; pdb.set_trace()
                raise RuntimeError("Non-matchable at {}-{}".format(i, j))

    FileUtils.save_file(map_dict, from_data_prefix + ".phrase.map")


def remap_retrieval(retrieval_out_prefix, query_map_dict_path, search_map_dict_path):
    n_src_shrads = len(list(glob(retrieval_out_prefix + "*.retrieval.out.pt")))
    query_map = FileUtils.load_file(query_map_dict_path)
    search_map = FileUtils.load_file(search_map_dict_path)
    logging.info("{} shards found".format(n_src_shrads))
    miss_query, miss_key = 0, 0
    total_query, total_key = 1e-9, 1e-9
    for i in range(n_src_shrads):
        retrieval_out_path = "{}.{}.retrieval.out.pt".format(retrieval_out_prefix, i)
        retrieval_out = FileUtils.load_file(retrieval_out_path)
        mapped_retrieval_out = {'query_ids': [], 'search_ids': [], 'search_dist': []}
        for q_id, s_ids, s_dist in zip(retrieval_out['query_ids'], retrieval_out['search_ids'], retrieval_out['search_dist']):
            total_query += 1
            if q_id in query_map:
                new_s_ids, new_s_dist = [], []
                for si, sd in zip(s_ids, s_dist):
                    total_key += 1
                    if si in search_map:
                        new_s_ids.append(search_map[si])
                        new_s_dist.append(sd)
                    else:
                        miss_key += 1
                mapped_retrieval_out['query_ids'].append(query_map[q_id])
                mapped_retrieval_out['search_ids'].append(new_s_ids)
                mapped_retrieval_out['search_dist'].append(new_s_dist)
            else:
                miss_query += 1
        FileUtils.save_file(retrieval_out, FileUtils.handle_file_extension(retrieval_out_path, "old"))
        FileUtils.save_file(mapped_retrieval_out, FileUtils.handle_file_extension(retrieval_out_path, "remap"))
        FileUtils.save_file(mapped_retrieval_out, retrieval_out_path)
    logging.info("{}/{} ({:.2f}%) queries missed".format(miss_query, total_query, miss_query/total_query * 100))
    logging.info("{}/{} ({:.2f}%) queries missed".format(miss_key, total_key, miss_key/total_key * 100))


def remap_pos(src_data_path, phrase_pos_prefix, retriever_tokenizer_path, mt_tokenizer_path, max_len=200):
    src_data = FileUtils.load_file(src_data_path)
    phrase_start_pos = FileUtils.load_file(phrase_pos_prefix + ".spos")
    phrase_end_pos = FileUtils.load_file(phrase_pos_prefix + ".epos")
    phrase_sid = FileUtils.load_file(phrase_pos_prefix + ".sid")
    phrase_repr = FileUtils.load_file(phrase_pos_prefix + ".npy")
    logging.info("Original num phrase: {}".format(phrase_repr.shape[0]))
    logging.info("Original num sent: {}".format(len(phrase_start_pos)))
    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_tokenizer_path)
    mt_tokenizer = FileUtils.load_spm(mt_tokenizer_path)
    ptr, repr_ptr = 0, 0
    new_phrase_start_pos = []
    new_phrase_end_pos = []
    new_phrase_repr_list = []
    new_phrase_rid_list = []
    new_phrase_sid = []
    for si, sent in tqdm(enumerate(src_data)):
        if si != phrase_sid[ptr]:
            print(si)
            continue
        retriever_tokenizer_out = retriever_tokenizer(sent, return_offsets_mapping=True, truncation=True)
        retriver_phrases, retriever_offset = retriever_tokenizer.convert_ids_to_tokens(retriever_tokenizer_out['input_ids']), retriever_tokenizer_out['offset_mapping']
        mt_phrases, mt_phrase_offset = DataUtils.sp_encode(mt_tokenizer, sent, return_offsets_mapping=True)
        offset_map = DataUtils.map_tokenized_sents(retriever_offset, mt_phrase_offset)
        new_starts, new_ends = [], []
        for s, e in zip(phrase_start_pos[ptr], phrase_end_pos[ptr]):
            if not offset_map[s] or not offset_map[e]:
                repr_ptr += 1
                logging.info("Empty offset at sent {}, original phrase".format(retriver_phrases[s:e+1]))
                continue
            if min(offset_map[s]) > max_len or max(offset_map[e]) > max_len:
                logging.info("Skip over max_len at sent {}, original phrase".format(retriver_phrases[s:e+1]))
                repr_ptr += 1
                continue
            new_phrase_repr_list.append(phrase_repr[repr_ptr])
            new_starts.append(min(offset_map[s]))
            new_ends.append(max(offset_map[e]))
            repr_ptr += 1
        new_phrase_rid_list.append(len(new_phrase_repr_list))
        new_phrase_sid.append(si)
        new_phrase_start_pos.append(new_starts)
        new_phrase_end_pos.append(new_ends)
        ptr += 1
    logging.info("New num phrase: {}".format(len(new_phrase_repr_list)))
    logging.info("New num sent: {}".format(len(new_phrase_start_pos)))
    FileUtils.save_file(new_phrase_sid, phrase_pos_prefix + ".mttok.sid")
    FileUtils.save_file(new_phrase_start_pos, phrase_pos_prefix + ".mttok.spos")
    FileUtils.save_file(new_phrase_end_pos, phrase_pos_prefix + ".mttok.epos")
    FileUtils.save_file(np.stack(new_phrase_repr_list, axis=0), phrase_pos_prefix + ".mttok.npy")
    FileUtils.save_file(new_phrase_rid_list, phrase_pos_prefix + ".mttok.rid")
        

def assemble_pbnmt_data_debuging_test(src_data_prefix, topk=5, save_type="npy", logging_step=320000, run_shards="", save_tag="debug"):
    n_src_shrads = len(list(glob(src_data_prefix + "*.repr.sid")))
    logging.info("{} src shards".format(n_src_shrads))
    # retrieval_out_data = FileUtils.load_shards(src_data_prefix, "retrieval.out.pt", n_src_shrads)
    src_phrase_sentid_list = FileUtils.load_shards(src_data_prefix, "repr.sid", n_src_shrads)
    src_phrase_start_pos_list = FileUtils.load_shards(src_data_prefix, "repr.spos", n_src_shrads)
    src_phrase_end_pos_list = FileUtils.load_shards(src_data_prefix, "repr.epos", n_src_shrads)
    src_phrase_repr_list = FileUtils.load_shards(src_data_prefix, "repr.dat", n_src_shrads)
    repr_list, repr_ids, spos_list, epos_list, dist_list, text_list, sid_list = [], [], [], [], [], [], []
    stats = dict()
    if run_shards:
        src_shard_ids = run_shards
        logging.info("Running on selected shards: {}".format(src_shard_ids))
    else:
        src_shard_ids = list(range(n_src_shrads))
        logging.info("Running on all shards: {}".format(src_shard_ids))
    prev_sentid = -1
    step = 0

    tmp_repr_list, tmp_spos_list, tmp_epos_list, tmp_dist_list = [], [], [], []
    tmp_text_list = set()

    for ptr in range(len(src_phrase_sentid_list)):
        sid = src_phrase_sentid_list[ptr]
        if sid != prev_sentid:
            if tmp_repr_list:
                sid_list.append(prev_sentid)
                repr_ids.append(len(tmp_repr_list) + len(repr_list))
                if len(tmp_repr_list) > stats.get("max_len_retrival", 0):
                    stats['max_len_retrival'] = len(tmp_repr_list)
                repr_list.extend(tmp_repr_list)
                spos_list.append(tmp_spos_list)
                epos_list.append(tmp_epos_list)
            prev_sentid = sid
            tmp_repr_list, tmp_spos_list, tmp_epos_list, tmp_dist_list = [], [], [], []

        tmp_repr_list.append(src_phrase_repr_list[ptr])
        tmp_spos_list.append(src_phrase_start_pos_list[ptr])
        tmp_epos_list.append(src_phrase_end_pos_list[ptr])

    if tmp_repr_list:
        sid_list.append(prev_sentid)
        repr_ids.append(len(tmp_repr_list) + len(repr_list))
        repr_list.extend(tmp_repr_list)
        spos_list.append(tmp_spos_list)
        epos_list.append(tmp_epos_list)
        
    if save_type == "pt":
        repr_data = torch.stack(repr_list)
        FileUtils.save_file(repr_data, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.pt")
    elif save_type == "npy":
        repr_data = torch.stack(repr_list)
        FileUtils.save_file(repr_data.numpy(), src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.npy")
    else:
        FileUtils.save_file(repr_list, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.list")

    FileUtils.save_file(sid_list, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.sid")
    FileUtils.save_file(repr_ids, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.rid")
    FileUtils.save_file(spos_list, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.spos")
    FileUtils.save_file(epos_list, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.epos")
    print("------------------ Stats ------------------")
    for k, v in stats.items():
        print("{}: {}".format(k, v))


def merge_pbnmt_data_shards(src_data_prefix, repr_type="npy", save_tag="all", enable_memmap=False):
    n_src_shrads = len(list(glob(src_data_prefix + "*.repr.sid")))
    logging.info("{}".format(src_data_prefix))
    logging.info("{} src shards found".format(n_src_shrads))
    repr_ids, spos_list, epos_list, dist_list, text_list, sid_list = [], [], [], [], [], []
    repr_data = None
    prev_repr_num = 0
    for src_shard_id in range(n_src_shrads):
        spos_list += FileUtils.load_file(src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.spos")
        epos_list += FileUtils.load_file(src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.epos")
        dist_list += FileUtils.load_file(src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.dist")
        text_list += FileUtils.load_file(src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.txt")
        sid_list += FileUtils.load_file(src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.sid")
        shard_repr_ids = FileUtils.load_file(src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.rid")
        shard_repr_ids = [j + prev_repr_num for j in shard_repr_ids]
        repr_ids += shard_repr_ids
        prev_repr_num = repr_ids[-1]
    h_num = repr_ids[-1]
    logging.info("{} reprs found".format(h_num))
    
    for src_shard_id in range(n_src_shrads):
        if repr_data is None:
            shard_repr_data = FileUtils.load_file(src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.{}".format(repr_type))
            prev_repr_num = 0
            if repr_type == "pt":
                shard_repr_data = shard_repr_data.numpy()
            elif repr_type != "npy":
                raise NotImplementedError
            shard_num, h_size = shard_repr_data.shape
            if enable_memmap:
                memmap_file = src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.memmap.npy"
                repr_data = np.memmap(memmap_file, dtype='float32', mode='w+', shape=(h_num, h_size))
                repr_data[prev_repr_num:prev_repr_num+shard_num] = shard_repr_data
            else:
                repr_data = shard_repr_data
            prev_repr_num += shard_num
        else:
            shard_repr_data = FileUtils.load_file(src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.{}".format(repr_type))
            if repr_type == "pt":
                shard_repr_data = shard_repr_data.numpy()
            elif repr_type != "npy":
                raise NotImplementedError
            shard_num = shard_repr_data.shape[0]
            if enable_memmap:
                repr_data[prev_repr_num:prev_repr_num+shard_num] = shard_repr_data
            else:
                repr_data = np.concatenate([repr_data, shard_repr_data], axis=0)
            prev_repr_num += shard_num
        logging.info("New prev_repr_num: {}".format(prev_repr_num))

    logging.info("repr_data shape: {}".format(repr_data.shape))
    if not enable_memmap:
        if repr_type == "npy":
            FileUtils.save_file(repr_data, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.{}".format(repr_type))
        elif repr_type == "pt":
            FileUtils.save_file(torch.from_numpy(repr_data), src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.{}".format(repr_type))
        else:
            raise NotImplementedError
    logging.info("sid_list size: {}".format(len(sid_list)))
    FileUtils.save_file(sid_list, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.sid")
    logging.info("repr_ids size: {}".format(len(repr_ids)))
    FileUtils.save_file(repr_ids, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.rid")
    logging.info("spos_list size: {}".format(len(spos_list)))
    FileUtils.save_file(spos_list, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.spos")
    logging.info("epos_list size: {}".format(len(epos_list)))
    FileUtils.save_file(epos_list, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.epos")
    logging.info("dist_list size: {}".format(len(dist_list)))
    FileUtils.save_file(dist_list, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.dist")
    logging.info("text_list size: {}".format(len(text_list)))
    FileUtils.save_file(text_list, src_data_prefix + ".{}".format(save_tag) + ".pbnmt.phrase.txt")


def convert_dat_to_memmap(data_prefix):
    n_shards = len(list(glob(data_prefix + "*.repr.dat")))
    logging.info("{} shards found".format(n_shards))
    memmap_file = "{}.memmap.npy".format(data_prefix)
    tgt_phrase_text_list = FileUtils.load_shards(data_prefix, "repr.txt", n_shards)
    repr_data = FileUtils.load_file("{}.{}.repr.dat".format(data_prefix, 0))
    h_size = repr_data[0].size(0)
    h_num = len(tgt_phrase_text_list)
    logging.info("{} phrases found".format(h_num))
    s, e = 0, len(repr_data)
    xb = np.memmap(memmap_file, dtype='float32', mode='w+', shape=(h_num, h_size))
    xb[s:e] = np.stack(repr_data, axis=0)
    s = e
    for i in range(1, n_shards):
        repr_data = FileUtils.load_file("{}.{}.repr.dat".format(data_prefix, i))
        e = s + len(repr_data)
        xb[s:e] = np.stack(repr_data, axis=0)
        s = e
    logging.info("{} reprs found".format(s))
    logging.info("Saving data to {}".format(memmap_file))
    

def assemble_pbnmt_data(src_data_prefix, tgt_data_prefix, topk=5, save_type="npy", logging_step=320000, run_shards="", min_dist=0, enable_memmap=False, retrieval_tag=""):
    n_src_shrads, n_tgt_shards = len(list(glob(src_data_prefix + "*.repr.sid"))), len(list(glob(tgt_data_prefix + "*.repr.dat")))
    logging.info("{} src shards and {} tgt shards".format(n_src_shrads, n_tgt_shards))
    if run_shards:
        src_shard_ids = run_shards
        logging.info("Running on selected shards: {}".format(src_shard_ids))
    else:
        src_shard_ids = list(range(n_src_shrads))
        logging.info("Running on all shards: {}".format(src_shard_ids))
    # retrieval_out_data = FileUtils.load_shards(src_data_prefix, "retrieval.out.pt", n_src_shrads)
    src_phrase_sentid_list = FileUtils.load_shards(src_data_prefix, "repr.sid", n_src_shrads)
    src_phrase_start_pos_list = FileUtils.load_shards(src_data_prefix, "repr.spos", n_src_shrads)
    src_phrase_end_pos_list = FileUtils.load_shards(src_data_prefix, "repr.epos", n_src_shrads)

    tgt_phrase_text_list = FileUtils.load_shards(tgt_data_prefix, "repr.txt", n_tgt_shards)
    tgt_phrase_sentid_list = FileUtils.load_shards(tgt_data_prefix, "repr.sid", n_tgt_shards)
    if enable_memmap:
        # please run convert_to_memmap before setting enable_memmap=True
        repr_data = FileUtils.load_file("{}.{}.repr.dat".format(tgt_data_prefix, 0))
        h_num, h_size = len(tgt_phrase_text_list), repr_data[0].size(0)
        memmap_file = "{}.memmap.npy".format(tgt_data_prefix)
        logging.info("Size of memmap tgt phrase data: {}".format((h_num, h_size)))
        tgt_phrase_repr_list = np.memmap(memmap_file, dtype='float32', mode='r', shape=(h_num, h_size))
    else:
        tgt_phrase_repr_list = FileUtils.load_shards(tgt_data_prefix, "repr.dat", n_tgt_shards)
    repr_list, repr_ids, spos_list, epos_list, dist_list, text_list, sid_list = [], [], [], [], [], [], []
    stats = dict()

    prev_sentid = -1
    step = 0

    for src_shard_id in src_shard_ids:
        if retrieval_tag:
            shard = FileUtils.load_file("{}.{}.{}.{}".format(src_data_prefix, src_shard_id, retrieval_tag, "retrieval.out.pt"))
        else:
            shard = FileUtils.load_file("{}.{}.{}".format(src_data_prefix, src_shard_id, "retrieval.out.pt"))
        query_ids, search_ids, search_dist = shard['query_ids'], shard['search_ids'], shard['search_dist']
        tmp_repr_list, tmp_spos_list, tmp_epos_list, tmp_dist_list = [], [], [], []
        tmp_text_list = set()

        for ptr in range(len(query_ids)):
            if (step + 1) % logging_step == 0:
                logging.info("Processed {} phrases".format(step+1))
            step += 1
            qid = query_ids[ptr]
            sid = src_phrase_sentid_list[qid]
            if sid != prev_sentid:
                if tmp_repr_list:
                    sid_list.append(prev_sentid)
                    repr_ids.append(len(tmp_repr_list) + len(repr_list))
                    if len(tmp_repr_list) > stats.get("max_len_retrival", 0):
                        stats['max_len_retrival'] = len(tmp_repr_list)
                    repr_list.extend(tmp_repr_list)
                    spos_list.append(tmp_spos_list)
                    epos_list.append(tmp_epos_list)
                    dist_list.append(tmp_dist_list)
                    text_list.append(" |**| ".join(list(tmp_text_list)))
                prev_sentid = sid
                tmp_repr_list, tmp_spos_list, tmp_epos_list, tmp_dist_list = [], [], [], []
                tmp_text_list = set()

            unique_tgt_phrase_ids, cur_dist = [], []
            for phrase_idx, phrase_dist in zip(search_ids[ptr], search_dist[ptr]):
                phrase_text = tgt_phrase_text_list[phrase_idx]
                phrase_sid = tgt_phrase_sentid_list[phrase_idx]
                if phrase_dist <= min_dist or phrase_text in tmp_text_list or phrase_sid == sid or len(unique_tgt_phrase_ids) >= topk:
                    continue
                tmp_text_list.add(phrase_text)
                unique_tgt_phrase_ids.append(phrase_idx)
                cur_dist.append(phrase_dist)
            if not unique_tgt_phrase_ids:
                continue
            cur_dist = torch.tensor(cur_dist).unsqueeze(-1)
            cur_dist = cur_dist / cur_dist.sum()
            if enable_memmap:
                cur_repr = torch.from_numpy(tgt_phrase_repr_list[unique_tgt_phrase_ids])
            else:
                cur_repr = torch.stack([tgt_phrase_repr_list[j].view(-1) for j in unique_tgt_phrase_ids])
            cur_repr = torch.sum(cur_repr * cur_dist, dim=0)
            tmp_repr_list.append(cur_repr)
            tmp_spos_list.append(src_phrase_start_pos_list[qid])
            tmp_epos_list.append(src_phrase_end_pos_list[qid])
            tmp_dist_list.append(cur_dist.tolist())

        if tmp_repr_list:
            sid_list.append(prev_sentid)
            repr_ids.append(len(tmp_repr_list) + len(repr_list))
            repr_list.extend(tmp_repr_list)
            spos_list.append(tmp_spos_list)
            epos_list.append(tmp_epos_list)
            dist_list.append(tmp_dist_list)
            text_list.append(" |**| ".join(list(tmp_text_list)))
        
        if save_type == "pt":
            repr_data = torch.stack(repr_list)
            FileUtils.save_file(repr_data, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.pt")
        elif save_type == "npy":
            repr_data = torch.stack(repr_list)
            FileUtils.save_file(repr_data.numpy(), src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.npy")
        else:
            FileUtils.save_file(repr_list, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.list")

        FileUtils.save_file(sid_list, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.sid")
        FileUtils.save_file(repr_ids, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.rid")
        FileUtils.save_file(spos_list, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.spos")
        FileUtils.save_file(epos_list, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.epos")
        FileUtils.save_file(dist_list, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.dist")
        FileUtils.save_file(text_list, src_data_prefix + ".{}".format(src_shard_id) + ".pbnmt.phrase.txt")
        repr_list, repr_ids, spos_list, epos_list, dist_list, text_list, sid_list = [], [], [], [], [], [], []

    print("------------------ Stats ------------------")
    for k, v in stats.items():
        print("{}: {}".format(k, v))

def extract_ngrams(input_file_path, encoder_tokenizer_path, save_prefix, max_ngram_len=4, min_pmi=0.00005, remove_top_freq=10000):
    def calculate_probs(c, total_num):
        return {w: f / total_num for w, f in c.most_common()}

    assert max_ngram_len % 2 == 0
    data = FileUtils.load_file(input_file_path)
    tokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_path)
    unigram_counter = Counter()
    tokenized_data = []
    total_unigram_num = 0
    for sent in tqdm(data):
        tokens = tokenizer.convert_ids_to_tokens(tokenizer([sent], truncation=True)['input_ids'][0][1:-1])
        tokenized_data.append(tokens)
        ngrams = DataUtils.extract_ngrams(tokens, 1)
        total_unigram_num += len(ngrams)
        unigram_counter.update(ngrams)
    unigram_prob = calculate_probs(unigram_counter, total_unigram_num)

    pmi_value_dict = dict()
    total_ngram_num, ngram_counter = 0, Counter()
    for tokens in tqdm(tokenized_data):
        ngrams = DataUtils.extract_ngrams(tokens, max_ngram_len)
        total_ngram_num += len(ngrams)
        ngram_counter.update(ngrams)
    ngram_prob = calculate_probs(ngram_counter, total_ngram_num)
    for key, prob in ngram_prob.items():
        word_probs = np.array([unigram_prob.get(w, 1e-9) for w in key])
        pmi_value_dict[key] = np.log(prob) - np.sum(np.log(word_probs))
                
    FileUtils.save_file({k: f for k, f in ngram_counter.most_common()}, save_prefix + ".{}.freq.pt".format(max_ngram_len))
    FileUtils.save_file(pmi_value_dict, save_prefix + ".{}.pmi.pt".format(max_ngram_len))
    FileUtils.save_file(ngram_prob, save_prefix + ".{}.pmi.pt".format(max_ngram_len))

# def extract_ngrams(input_file_path, save_path, max_ngram_len=3, min_freq=5):
#     ngram_lens = list(range(1, max_ngram_len+1))
#     logging.info("Ngram Lengths: {}".format(ngram_lens))
#     data = FileUtils.load_file(input_file_path)
#     lang = FileUtils.check_file_type(input_file_path)
#     moses_tokenizer = DataUtils.get_moses_tokenizer(lang)
#     assert lang in LANGUAGE_LIST
#     counter = Counter()
#     for sent in tqdm(data):
#         words = moses_tokenizer.tokenize(sent, escape=False)
#         for n in ngram_lens:
#             counter.update(DataUtils.extract_ngrams(words, n))
#     ngram_set = set()
#     ngram_nums_before = {k: 0 for k in ngram_lens}
#     ngram_nums_after = {k: 0 for k in ngram_lens}
#     for k, v in counter.most_common():
#         ngram_nums_before[len(k)] += 1
#         if v < min_freq:
#             continue
#         else:
#             ngram_nums_after[len(k)] += 1
#             ngram_set.add(k)
#     logging.info("Nums of ngrams before filter {}".format(ngram_nums_before))
#     logging.info("Nums of ngrams after filter {}".format(ngram_nums_after))
#     FileUtils.save_file(ngram_set, save_path, "pt")


def encode_phrase(encoder_model_path,
                  encoder_tokenizer_path,
                  phrase_tokenizer_path,
                  input_file_path,
                  save_pref,
                  batch_size=64,
                  cache_size=50000,
                  layer=-1,
                  use_sent_repr=False):

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    encoder = AutoModel.from_pretrained(encoder_model_path).eval().to(device)
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_path)
    phrase_tokenizer = FileUtils.load_spm(phrase_tokenizer_path)
    data = FileUtils.load_file(input_file_path)
    batch, indices, repr_list, phrase_list = [], [], [], []
    n_repr = 0
    FileUtils.check_dirs(FileUtils.get_dir(save_pref))
    n_shard = 0

    for line in data:
        batch.append(line.strip())
        if len(batch) >= batch_size:
            encoder_batch = encoder_tokenizer(batch, return_offsets_mapping=True, truncation=True)
            _, encoder_batch_offset = encoder_batch['input_ids'], encoder_batch['offset_mapping']
            phrase_batch_data, phrase_batch_offset = DataUtils.sp_encode(phrase_tokenizer, batch, return_offsets_mapping=True)
            outputs, _ = HFUtils.encode(encoder, encoder_tokenizer, batch, device)
            outputs = outputs.hidden_states[layer].cpu()
            cls_repr = outputs[:, 0]
            offset_map = [DataUtils.map_tokenized_sents(po, eo) for eo, po in zip(encoder_batch_offset, phrase_batch_offset)]
            for batch_id, (sent, sent_map) in enumerate(zip(phrase_batch_data, offset_map)):
                for p, p_ids in zip(sent, sent_map):
                    if len(p.split("▁")) > 2 and p_ids:
                        s, e = p_ids[0], p_ids[-1]
                        phrase_repr = outputs[batch_id, e] - outputs[batch_id, s]
                        if use_sent_repr:
                            phrase_repr = torch.cat([cls_repr[batch_id], phrase_repr], dim=-1)
                        indices.append(n_repr)
                        repr_list.append(phrase_repr)
                        phrase_list.append(p)
                        n_repr += 1
            batch = []
            if len(indices) >= cache_size:
                FileUtils.save_to_disk(indices, save_pref + ".{}.repr.idx".format(n_shard))
                FileUtils.save_to_disk(repr_list, save_pref + ".{}.repr.dat".format(n_shard))
                indices = []
                repr_list = []
                n_shard += 1
        
    if batch:
        encoder_batch = encoder_tokenizer(batch, return_offsets_mapping=True, truncation=True)
        _, encoder_batch_offset = encoder_batch['input_ids'], encoder_batch['offset_mapping']
        phrase_batch_data, phrase_batch_offset = DataUtils.sp_encode(phrase_tokenizer, batch, return_offsets_mapping=True)
        outputs, _ = HFUtils.encode(encoder, encoder_tokenizer, batch, device)
        outputs = outputs.hidden_states[layer].cpu()
        cls_repr = outputs[:, 0]
        offset_map = [DataUtils.map_tokenized_sents(po, eo[1:-1]) for eo, po in zip(encoder_batch_offset, phrase_batch_offset)]
        for batch_id, (sent, sent_map) in enumerate(zip(phrase_batch_data, offset_map)):
            for p, p_ids in zip(sent, sent_map):
                if len(p.split("▁")) > 2 and p_ids:
                    s, e = p_ids[0], p_ids[-1]+1
                    phrase_repr = outputs[batch_id, e] - outputs[batch_id, s]
                    if use_sent_repr:
                        phrase_repr = torch.cat([cls_repr[batch_id], phrase_repr], dim=-1)
                    indices.append(n_repr)
                    repr_list.append(phrase_repr)
                    phrase_list.append(p)
                    n_repr += 1
        batch = []

    FileUtils.save_to_disk(indices, save_pref + ".{}.repr.idx".format(n_shard))
    FileUtils.save_to_disk(repr_list, save_pref + ".{}.repr.dat".format(n_shard))
    FileUtils.save_to_disk(phrase_list, save_pref + ".repr.txt")


def sp_encode_file(input_files, model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    model_pref = ".".join(path.basename(model_path).split(".")[:-1])
    for fname in input_files.split(";"):
        bpe_data = []
        logging.info(f"Processing {fname}")
        with open(fname) as fin:
            for line in tqdm(fin):
                subwords = DataUtils.sp_encode(sp, line.strip())
                subwords = " ".join(subwords)
                bpe_data.append(subwords)
            logging.info("{} sents".format(len(bpe_data)))
            FileUtils.save_to_disk(bpe_data, FileUtils.handle_file_extension(fname, model_pref))


def fastbpe_encode_file(input_file, codes_path, vocab_path):
    import fastBPE
    bpe = fastBPE.fastBPE(codes_path, vocab_path)

    bpe_data = []
    logging.info(f"Processing {input_file}")
    with open(input_file) as fin:
        data = []
        for line in tqdm(fin):
            data.append(line)
        bpe_data = bpe.apply(data)
    FileUtils.save_to_disk(bpe_data, FileUtils.handle_file_extension(input_file, "fastbpe"))


def extract_subset(path_to_ids, src_file, tgt_file, tag="rand"):
    src_data = FileUtils.load_file(src_file, file_type=FileType.TXT)
    tgt_data = FileUtils.load_file(tgt_file, file_type=FileType.TXT)
    ids = FileUtils.load_file(path_to_ids, file_type=FileType.PT)
    ids = sorted(list(ids))
    new_src_data, new_tgt_data = [], []
    for i in ids:
        new_src_data.append(src_data[i])
        new_tgt_data.append(tgt_data[i])
    FileUtils.save_to_disk(new_src_data, FileUtils.handle_file_extension(src_file, tag))
    FileUtils.save_to_disk(new_tgt_data, FileUtils.handle_file_extension(tgt_file, tag))


def ngram_sent_matching(query_file, data_file, n=3):
    ngram2sent = dict()
    with open(data_file) as fin:
        logging.info("Building ngram mapping...")
        for lno, line in enumerate(tqdm(fin)):
            words = line.strip().split()
            ngrams = DataUtils.extract_ngrams(words, n)
            for ng in ngrams:
                if ng in ngram2sent:
                    ngram2sent[ng].append(lno)
                else:
                    ngram2sent[ng] = [lno]
    cov_num, total_num = 0, 0
    with open(query_file) as fin:
        return_data = []
        logging.info("Querying...")
        for lno, line in enumerate(tqdm(fin)):
            words = line.strip().split()
            ngrams = DataUtils.extract_ngrams(words, n)
            sent_data = []
            for ng in ngrams:
                if ng in ngram2sent:
                    sent_data.append(ngram2sent[ng])
                    cov_num += 1
                else:
                    sent_data.append([])
                total_num += 1
            return_data.append(sent_data)
    logging.info("{:.2f}% ngrams in query file are covered...".format(cov_num/total_num*100))
    save_path = FileUtils.handle_file_extension(query_file, "cov.{}".format(n))
    # logging.info(f"Saving data to {save_path}")
    # Utils.save_to_disk(return_data, save_path, file_type=FileType.PT)


def wmt_to_txt(hfds_path, save_dir, src_lang="de", tgt_lang="en", subsets="train;validation;test"):
    from datasets import load_from_disk
    ds = load_from_disk(hfds_path)
    if not path.exists(save_dir):
        logging.warning("No directory found... Making by myself")
        os.mkdir(save_dir)
    for sub in subsets.split(";"):
        logging.info("Processing {}".format(sub))
        src_data, tgt_data = [], []
        for it in tqdm(ds[sub]):
            src_data.append(it['translation'][src_lang])
            tgt_data.append(it['translation'][tgt_lang])
        FileUtils.save_to_disk(src_data, save_dir+"/{}.{}".format(sub, src_lang))
        FileUtils.save_to_disk(tgt_data, save_dir+"/{}.{}".format(sub, tgt_lang))


def remove_duplicate_text(input_file):
    data = FileUtils.load_file(input_file)
    FileUtils.save_to_disk(DataUtils.remove_duplicate_text(data), FileUtils.handle_file_extension(input_file, "uniq", 'add'))


def build_phrase2sent_index(file_path, phrase_tokenizer_path, subword_tokenizer_path, save_path, batch_size=64, max_sent_num=64000):
    data = FileUtils.load_file(file_path)
    phrase_tokenizer = FileUtils.load_spm(phrase_tokenizer_path)
    subword_tokenizer = FileUtils.load_spm(subword_tokenizer_path)
    rand_data_ids = list(range(len(data)))
    random.shuffle(rand_data_ids)
    def is_phrase(piece, subword_tokenizer):
        return subword_tokenizer.piece_to_id(piece) == subword_tokenizer.unk_id()
    
    s, data_size = 0, len(data)
    phrase2sent_dict = dict()
    while s < data_size:
        e = min(s + batch_size, data_size)
        batch_ids = rand_data_ids[s:e]
        batch = [data[i] for i in batch_ids]
        tok_batch = DataUtils.sp_encode(phrase_tokenizer, batch)
        for sent_id, sent_tok_list in zip(batch_ids, tok_batch):
            for p in sent_tok_list:
                if is_phrase(p, subword_tokenizer):
                    if p in phrase2sent_dict and len(phrase2sent_dict[p]) < max_sent_num:
                        phrase2sent_dict[p].append(sent_id)
                    else:
                        phrase2sent_dict[p] = [sent_id]
        s = e
    FileUtils.save_to_disk(phrase2sent_dict, save_path)


def extract_moses_phrase_table(src_file_path, tgt_file_path, s2t_symal_path, save_path, src_lang, tgt_lang, min_freq=5):
    orig_src_data = FileUtils.load_file(src_file_path)
    orig_tgt_data = FileUtils.load_file(tgt_file_path)
    src_lang, tgt_lang = FileUtils.check_file_type(src_file_path), FileUtils.check_file_type(tgt_file_path)
    assert src_lang in LANGUAGE_LIST and tgt_lang in LANGUAGE_LIST
    src_data, tgt_data, s2t_align_data = FileUtils.load_symal(s2t_symal_path, return_str=True)
    phrase_table = Counter()
    n_invalid = 0

    def check_index_range(src_words, tgt_words, s2t):
        s_ids = [it[0] for it in s2t]
        t_ids = [it[1] for it in s2t]
        min_s_ids, max_s_ids = min(s_ids), max(s_ids)
        min_t_ids, max_t_ids = min(t_ids), max(t_ids)
        n_s_words, n_t_words = len(src_words), len(tgt_words)
        if min_s_ids < 0 or max_s_ids >= n_s_words or min_t_ids < 0 or max_t_ids >= n_t_words:
            return False
        else:
            return True
    
    def check_adjacent(ui, vi):
        return ui == vi or ui == (vi+1) or (ui+1) == vi
    
    src_moses_tokenizer = DataUtils.get_moses_tokenizer(src_lang)
    tgt_moses_tokenizer = DataUtils.get_moses_tokenizer(tgt_lang)

    for sent_id, (orig_src, orig_tgt, src, tgt, s2t) in enumerate(zip(orig_src_data, orig_tgt_data, src_data, tgt_data, s2t_align_data)):
        orig_src = src_moses_tokenizer.tokenize(orig_src, return_str=True)
        orig_tgt = tgt_moses_tokenizer.tokenize(orig_tgt, return_str=True)
        if len(orig_src.split()) != len(src.split()) or len(orig_tgt.split()) != len(tgt.split()):
            logging.warn("Skip length-changed sentence {}...".format(sent_id))
            continue
        src_words = orig_src.split()
        tgt_words = orig_tgt.split()
        if not check_index_range(src_words, tgt_words, s2t):
            n_invalid += 1
            logging.warn("Skip invalid sentence {}...".format(sent_id))
            continue
        n_s2t = len(s2t)
        for i in range(n_s2t):
            cur_sw_id, cur_tw_id = s2t[i]
            src_phrase, tgt_phrase = src_words[cur_sw_id], tgt_words[cur_tw_id]
            tmp_sw_id, tmp_tw_id = cur_sw_id, cur_tw_id
            phrase_table[(src_phrase, tgt_phrase)] += 1
            for j in range(i+1, n_s2t):
                next_sw_id, next_tw_id = s2t[j]
                if check_adjacent(tmp_sw_id, next_sw_id) and check_adjacent(tmp_tw_id, next_tw_id):
                    src_phrase = " ".join(src_words[cur_sw_id:next_sw_id+1])
                    tgt_phrase = " ".join(tgt_words[cur_tw_id:next_tw_id+1])
                    phrase_table[(src_phrase, tgt_phrase)] += 1
                    tmp_sw_id, tmp_tw_id = next_sw_id, next_tw_id
                else:
                    break
    final_phrase_table = []
    for key, freq in phrase_table.most_common():
        if freq < min_freq:
            continue
        final_phrase_table.append(key + (freq,))

    FileUtils.save_to_disk(final_phrase_table, save_path)


def extract_sp_phrase_table(src_file_path, tgt_file_path, s2t_symal_path, src_phrase_tokenizer_path, tgt_phrase_tokenizer_path, subword_tokenizer_path, save_path, min_freq=3):
    """
    Note: src file and tgt file are un-tokenized data
    Goal: we plan to find the aligned target phrases of each src words/phrase.
    """
    src_data = FileUtils.load_file(src_file_path)
    tgt_data = FileUtils.load_file(tgt_file_path)
    src_lang, tgt_lang = FileUtils.check_file_type(src_file_path), FileUtils.check_file_type(tgt_file_path)
    assert src_lang in LANGUAGE_LIST and tgt_lang in LANGUAGE_LIST
    align_src_lens, align_tgt_lens, s2t_align_data = FileUtils.load_symal(s2t_symal_path)
    src_moses_tokenizer = DataUtils.get_moses_tokenizer(src_lang)
    tgt_moses_tokenizer = DataUtils.get_moses_tokenizer(tgt_lang)
    src_phrase_tokenizer = FileUtils.load_spm(src_phrase_tokenizer_path)
    tgt_phrase_tokenizer = FileUtils.load_spm(tgt_phrase_tokenizer_path)
    subword_tokenizer = FileUtils.load_spm(subword_tokenizer_path)
    phrase_table = Counter()
    n_invalid = 0
    max_num_punc_rate = 0.5

    def is_phrase(piece, subword_tokenizer):
        # True means the piece is unseen in subword_tokenizer. Thus, it will be 
        # split into multiple parts, i.e. a phrase.
        return subword_tokenizer.piece_to_id(piece) == subword_tokenizer.unk_id()
    
    def cov_rate(goal_piece_offset, aux_piece_offset_list, str_len):
        s1, e1 = goal_piece_offset
        cov_list = [0] * str_len
        for s2, e2 in aux_piece_offset_list:
            for j in range(s2, e2):
                if s1 <= j < e1:
                    cov_list[j] = 1
        return sum(cov_list) / (e1 - s1)
    
    def find_skip_set(tgt_char_len, tgt_moses_ids, tgt_moses_offset, tgt_moses2phrase_map, tgt_phrase_offset, min_cov_rate=0.6):
        # skip phrases that are not well covered
        aux_piece_offset_list = [tgt_moses_offset[i] for i in tgt_moses_ids]
        skip_set = set()
        prev = -1
        for tid in tgt_moses_ids:
            for pid in tgt_moses2phrase_map[tid]:
                if pid == prev:
                    continue
                prev = pid
                goal_piece_offset = tgt_phrase_offset[pid]
                cr = cov_rate(goal_piece_offset, aux_piece_offset_list, tgt_char_len)
                if cr < min_cov_rate:
                    skip_set.add(pid)
        return skip_set
        
    def update_phrase(phrase_table, last_src_phrase_str, tgt_moses_ids, tgt_moses2phrase_map, tgt_phrase_tok_list, skip_set=set()):
        prev_idx = -1
        for i in tgt_moses_ids:
            for j in tgt_moses2phrase_map[i]:
                if j == prev_idx or j in skip_set:
                    continue
                prev_idx = j
                if is_phrase(tgt_phrase_tok_list[j], subword_tokenizer) and DataUtils.num_and_punc_rate(tgt_phrase_tok_list[j]) < max_num_punc_rate:
                    phrase_table[(last_src_phrase_str, tgt_phrase_tok_list[j])] += 1

    for sent_id, (src, tgt, al_src_len, al_tgt_len, s2t) in enumerate(zip(src_data, tgt_data, align_src_lens, align_tgt_lens, s2t_align_data)):
        try:
            src_moses_tok_sent = src_moses_tokenizer.tokenize(src, escape=False, return_str=True)
            tgt_moses_tok_sent = tgt_moses_tokenizer.tokenize(tgt, escape=False, return_str=True)
            src_moses_offset = DataUtils.moses_offset(src, src_moses_tok_sent)
            tgt_moses_offset = DataUtils.moses_offset(tgt, tgt_moses_tok_sent)
            src_phrase_tok_list, src_phrase_offset = DataUtils.sp_encode(src_phrase_tokenizer, src, return_offsets_mapping=True)
            tgt_phrase_tok_list, tgt_phrase_offset = DataUtils.sp_encode(tgt_phrase_tokenizer, tgt, return_offsets_mapping=True)
            src_moses2phrase_map = DataUtils.map_tokenized_sents(src_moses_offset, src_phrase_offset)
            tgt_moses2phrase_map = DataUtils.map_tokenized_sents(tgt_moses_offset, tgt_phrase_offset)
            if al_src_len == 0 or al_tgt_len == 0 or len(src_moses_tok_sent.split()) != al_src_len or len(tgt_moses_tok_sent.split()) != al_tgt_len or not src_moses_offset or not tgt_moses_offset:
                n_invalid += 1
                logging.warn("Skip sentence {}...".format(sent_id))
                continue

            tgt_char_len = len(tgt)
            src_moses_ids, tgt_moses_ids = [], []
            for s2t_ptr in range(len(s2t)):
                sid, tid = s2t[s2t_ptr]
                # is_src_side_phrase = is_phrase(src[raw_s:raw_e], subword_tokenizer)
                if not src_moses_ids:
                    # if is_src_side_phrase:
                    if len(src_moses2phrase_map[sid]) == 1:
                        src_moses_ids.append(sid)
                        tgt_moses_ids.append(tid)
                else:
                    # if len(src_moses2phrase_map[sid]) > 1 or not is_src_side_phrase:
                    if len(src_moses2phrase_map[sid]) > 1:
                        # This branch means that we meet a word that was tokenized to several parts by the phrase tokenizer.
                        last_src_phrase_id = src_moses2phrase_map[src_moses_ids[-1]][-1]
                        last_src_phrase_str = src_phrase_tok_list[last_src_phrase_id]
                        if DataUtils.num_and_punc_rate(last_src_phrase_str) < max_num_punc_rate:
                            if last_src_phrase_id in src_moses2phrase_map[sid]:
                                tgt_moses_ids.append(tid)
                            # finalize current phrase
                            cur_skip_set = find_skip_set(tgt_char_len, tgt_moses_ids, tgt_moses_offset, tgt_moses2phrase_map, tgt_phrase_offset)
                            update_phrase(phrase_table, last_src_phrase_str, tgt_moses_ids, tgt_moses2phrase_map, tgt_phrase_tok_list, skip_set=cur_skip_set)
                        src_moses_ids, tgt_moses_ids = [], []
                    elif src_moses2phrase_map[sid][-1] != src_moses2phrase_map[src_moses_ids[-1]][0]:
                        # This branch means that we are visiting a new src phrase
                        # finalize current phrase
                        last_src_phrase_id = src_moses2phrase_map[src_moses_ids[-1]][-1]
                        last_src_phrase_str = src_phrase_tok_list[last_src_phrase_id]
                        if DataUtils.num_and_punc_rate(last_src_phrase_str) < max_num_punc_rate:
                            cur_skip_set = find_skip_set(tgt_char_len, tgt_moses_ids, tgt_moses_offset, tgt_moses2phrase_map, tgt_phrase_offset)
                            update_phrase(phrase_table, last_src_phrase_str, tgt_moses_ids, tgt_moses2phrase_map, tgt_phrase_tok_list, skip_set=cur_skip_set)
                        src_moses_ids, tgt_moses_ids = [sid], [tid]
                    else:
                        # This branch means that we are still on the same phrase
                        src_moses_ids.append(sid)
                        tgt_moses_ids.append(tid)

            if src_moses_ids and tgt_moses_ids:
                last_src_phrase_id = src_moses2phrase_map[src_moses_ids[-1]][-1]
                last_src_phrase_str = src_phrase_tok_list[last_src_phrase_id]
                if DataUtils.num_and_punc_rate(last_src_phrase_str) < max_num_punc_rate:
                    cur_skip_set = find_skip_set(tgt_char_len, tgt_moses_ids, tgt_moses_offset, tgt_moses2phrase_map, tgt_phrase_offset)
                    update_phrase(phrase_table, last_src_phrase_str, tgt_moses_ids, tgt_moses2phrase_map, tgt_phrase_tok_list, skip_set=cur_skip_set)
        except Exception:
            logging.error("Sentence {} meets runtime error...".format(sent_id))
    # frequency filter
    final_phrase_table = []
    for key, freq in phrase_table.most_common():
        if freq < min_freq:
            continue
        final_phrase_table.append(key + (freq,))

    FileUtils.save_to_disk(final_phrase_table, save_path)


def train_sp_model(input_file, model_prefix="bpe_model", vocab_size=1000, model_type="bpe"):
    # Train the SentencePiece BPE model
    spm.SentencePieceTrainer.train(
        input=input_file,  # Input sentences
        model_prefix=model_prefix,  # Prefix for model files (bpe_model.model, bpe_model.vocab)
        vocab_size=vocab_size,  # Size of the vocabulary
        model_type=model_type,  # Model type: "bpe", "unigram", "word", or "char"
    )

    print(f"Model trained successfully with model_prefix: '{model_prefix}', vocab_size: {vocab_size}, model_type: '{model_type}'")


def train_sp_model_for_longpiece(input_file, model_prefix="320k.bpe", vocab_size=320000, model_type="bpe", max_piece_len=36):
    # Train the SentencePiece BPE model
    logging.info("Start training BPE model for long pieces")
    spm.SentencePieceTrainer.train(
        input=input_file,  # Input sentences
        model_prefix=model_prefix,  # Prefix for model files (bpe_model.model, bpe_model.vocab)
        vocab_size=vocab_size,  # Size of the vocabulary
        model_type=model_type,  # Model type: "bpe", "unigram", "word", or "char"
        max_sentencepiece_length=max_piece_len,
        split_by_unicode_script=True,
        split_by_number=True,
        split_by_whitespace=False,
        num_threads=32
    )
    logging.info(f"Model trained successfully with model_prefix: '{model_prefix}', vocab_size: {vocab_size}, model_type: '{model_type}'")


def encode_decode_example(model_prefix="bpe_model"):
    # Load the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    # Encode a sentence using the trained model
    encoded = sp.encode("This is an encoded example sentence.", out_type=str)
    print("Encoded sentence:", encoded)

    # Decode the encoded sentence back to the original text
    decoded = sp.decode(encoded)
    print("Decoded sentence:", decoded)


def extract_phrases(vocab_file, lang_full_name):
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    mystopwords = stopwords.words(lang_full_name)
    vocab_data = [it.split("\t")[0] for it in FileUtils.load_file(vocab_file, file_type=FileType.TXT)]
    phrases = []
    useful_phrases = []
    n_useful = 0
    for it in vocab_data:
        pieces = it.lower().split("▁")
        if pieces[0] == "":
            pieces = pieces[1:]
        if len(pieces) >= 2:
            s, e = 0, len(pieces)-1
            for i in range(len(pieces)):
                if pieces[i] in mystopwords:
                    s += 1
                else:
                    break
            for i in range(len(pieces)-1, -1, -1):
                if pieces[i] in mystopwords:
                    e -= 1
                else:
                    break
            if (e - s) >= 1:
                useful_phrases.append(it)
                n_useful += 1
            phrases.append(it)
    logging.info("total phrase num {}\tvalid phrase num {}\tpercent {:.2f}%".format(len(phrases), n_useful, n_useful/len(phrases)*100))
    FileUtils.save_to_disk(phrases, vocab_file + ".phrase", 'txt')
    FileUtils.save_to_disk(useful_phrases, vocab_file + ".useful.phrase", 'txt')


def tokenized_data_analysis(input_file, vocab_file, lang_full_name):
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    mystopwords = stopwords.words(lang_full_name)
    vocab_data = [it.split("\t")[0] for it in FileUtils.load_file(vocab_file, file_type=FileType.TXT)]
    phrases = []
    useful_phrases = []
    n_useful = 0
    for it in vocab_data:
        pieces = it.lower().split("▁")
        if pieces[0] == "":
            pieces = pieces[1:]
        if len(pieces) >= 2:
            s, e = 0, len(pieces)-1
            for i in range(len(pieces)):
                if pieces[i] in mystopwords:
                    s += 1
                else:
                    break
            for i in range(len(pieces)-1, -1, -1):
                if pieces[i] in mystopwords:
                    e -= 1
                else:
                    break
            if (e - s) >= 1:
                useful_phrases.append(it)
                n_useful += 1
            phrases.append(it)
    phrases = set(phrases)
    useful_phrases = set(useful_phrases)
    input_data = FileUtils.load_file(input_file, file_type=FileType.TXT)
    input_words = []
    for it in input_data:
        input_words += it.split()
    n_words = len(input_words)
    n_phrase = 0
    n_useful = 0
    useful_phrases_in_data, phrases_in_data = [], []
    for w in input_words:
        if w in phrases:
            n_phrase += 1
            phrases_in_data.append(w)
        if w in useful_phrases:
            n_useful += 1
            useful_phrases_in_data.append(w)
    logging.info("Num of words & phrase: {}".format(n_words))
    logging.info("Num of phrase: {}".format(n_phrase))
    logging.info("Num of useful phrase: {}".format(n_useful))
    FileUtils.save_to_disk(phrases_in_data, "{}.phrase".format(input_file))
    FileUtils.save_to_disk(useful_phrases_in_data, "{}.useful.phrase".format(input_file))


def split_retrieval_output(retrieval_output_path, data_source_file_path, save_path, src_lang, tgt_lang, field="train"):
    retrieval_output = FileUtils.load_file(retrieval_output_path, 'json')
    data_source = FileUtils.load_file(data_source_file_path, 'txt')
    if field == "train":
        field = "sample.{}{}".format(src_lang, tgt_lang)
    elif field == "validation":
        field = "validation.{}{}".format(src_lang, tgt_lang)
    elif field == "test":
        field = "test.{}{}".format(src_lang, tgt_lang)
    else:
        raise ValueError
    start_case_id = 0
    for i, it in enumerate(data_source):
        if it == field:
            start_case_id = i
            break
    visited = False
    new_retrieval_output = []
    for case in retrieval_output:
        case_id = case['id']
        if data_source[case_id] == field:
            case['id'] = case_id - start_case_id
            new_retrieval_output.append(case)
            visited = True
        elif visited:
            break

    logging.info("Found {} examples for {}".format(len(new_retrieval_output), field))
    FileUtils.save_file(new_retrieval_output, save_path)


def merge_alpaca_data(data_path_list, save_path):
    file_names = data_path_list.split(",")
    data = None
    data_field = None
    for fname in file_names:
        cur_data = FileUtils.load_file(fname, 'json')
        if data is None:
            data = cur_data
            data_field = list(data.keys())[0]
        else:
            cur_data_field = list(cur_data.keys())[0]
            assert data_field == cur_data_field
            data[data_field].extend(cur_data[cur_data_field])
    logging.info("{} examples merged".format(len(data[data_field])))
    FileUtils.save_file(data, save_path)


def aligned_pair(data_dir_prefix, lang_list):
    num = 0
    for lang in lang_list:
        align_data = FileUtils.load_file("{}.{}en/alignment.pt".format(data_dir_prefix, lang))
        lang_pair_num = 0
        for it in align_data:
            lang_pair_num += len(it)
        logging.info("Found {} pairs for {}-en".format(lang_pair_num, lang))
        num += lang_pair_num
    print("Found {} pairs in total".format(num))


def sample_annotated_data(xlsx_path, topk=200, tag="", col_n=3):
    data = FileUtils.load_file(xlsx_path)
    indices = []
    output_path = FileUtils.handle_file_extension(xlsx_path, ".pt", "change")
    if tag:
        output_path = FileUtils.handle_file_extension(output_path, tag, "add")
    for i, it in enumerate(data):
        if it[col_n] == 1:
            indices.append(i-1)
    logging.info("{} indices found".format(len(indices)))
    indices = indices[:topk]
    FileUtils.save_file(indices, output_path)


def main():
    fire.Fire({
        "fastbpe_encode_file": fastbpe_encode_file,
        "extract_subset": extract_subset,
        "ngram_sent_matching": ngram_sent_matching,
        "sp_encode_file": sp_encode_file,
        "wmt_to_txt": wmt_to_txt,
        "train_sp_model": train_sp_model,
        "train_sp_model_for_longpiece": train_sp_model_for_longpiece,
        "extract_moses_phrase_table": extract_moses_phrase_table,
        "extract_phrases": extract_phrases,
        "tokenized_data_analysis": tokenized_data_analysis,
        "encode_phrase": encode_phrase,
        "remove_duplicate_text": remove_duplicate_text,
        "extract_sp_phrase_table": extract_sp_phrase_table,
        "build_phrase2sent_index": build_phrase2sent_index,
        "extract_ngrams": extract_ngrams,
        "assemble_pbnmt_data": assemble_pbnmt_data,
        "remap_pos": remap_pos,
        "assemble_pbnmt_data_debuging_test": assemble_pbnmt_data_debuging_test,
        "analyze_retrieval_dist": analyze_retrieval_dist,
        "merge_pbnmt_data_shards": merge_pbnmt_data_shards,
        "remap_phrase": remap_phrase,
        "remap_retrieval": remap_retrieval,
        "mix_pbnmt_data": mix_pbnmt_data,
        "convert_dat_to_memmap": convert_dat_to_memmap,
        "assemble_chunk_data": assemble_chunk_data,
        "test_tokenizer_time": test_tokenizer_time,
        "tokenize_corpus": tokenize_corpus,
        "test_retrieval_after_addvalid": test_retrieval_after_addvalid,
        "assemble_analysis_data": assemble_analysis_data,
        "merge_analysis_data": merge_analysis_data,
        "split_retrieval_output": split_retrieval_output,
        "merge_alpaca_data": merge_alpaca_data,
        "assemble_sentence_retrieval_data": assemble_sentence_retrieval_data,
        "aligned_pair": aligned_pair,
        "sample_annotated_data": sample_annotated_data
    })

if __name__ == "__main__":
    main()
