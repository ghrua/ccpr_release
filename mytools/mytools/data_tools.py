import sentencepiece as spm
import fire
import json
import os
import os.path as path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from collections import Counter
from mytools.tool_utils import FileUtils, DataUtils, LANGUAGE_LIST, logging
from tempfile import mkdtemp
import faiss
from faiss.contrib.ondisk import merge_ondisk
import numpy as np
import random
random.seed(10086)


def moses_tokenization(input_file, lower=False, escape=False):
    lang = FileUtils.check_file_type(input_file)
    moses_tokenizer = DataUtils.get_moses_tokenizer(lang)
    logging.info("Got the Moses tokenizer for {}".format(lang))
    tk_data = []
    raw_data = FileUtils.load_file(input_file)
    if lower:
        for line in tqdm(raw_data):
            moses_tok_sent = moses_tokenizer.tokenize(line.strip(), escape=escape, return_str=True)
            tk_data.append(moses_tok_sent.lower())
    else:
        for line in tqdm(raw_data):
            moses_tok_sent = moses_tokenizer.tokenize(line.strip(), escape=escape, return_str=True)
            tk_data.append(moses_tok_sent)
    file_tag = "tk"
    if lower:
        file_tag += ".lw"
    if escape:
        file_tag += ".ep"
    FileUtils.save_file(tk_data, FileUtils.handle_file_extension(input_file, file_tag))


def build_word_vocab(input_file, save_path):
    lang = FileUtils.check_file_type(input_file)
    if lang not in LANGUAGE_LIST:
        logging.error("Unkown language: {}".format(lang))
        raise ValueError
    
    moses_tokenizer = DataUtils.get_moses_tokenizer(lang)
    data = FileUtils.load_file(input_file)
    c = Counter()
    for line in data:
        words = moses_tokenizer.tokenize(line, escape=False)
        c.update(words)
    tk_data = []
    for k, v in c.most_common():
        tk_data.append("{}\t{}".format(k, v))
    FileUtils.save_file(tk_data, save_path)


def sample_from_bilingual_data(src_input_file, tgt_input_file, num_samples=2000, file_type="txt"):
    src_data = FileUtils.load_file(src_input_file, file_type=file_type)
    tgt_data = FileUtils.load_file(tgt_input_file, file_type=file_type)
    data_size = len(src_data)
    ids = list(range(data_size))
    sampled_ids = random.sample(ids, k=num_samples)
    sampled_src_data = [src_data[i] for i in sampled_ids]
    sampled_tgt_data = [tgt_data[i] for i in sampled_ids]
    FileUtils.save_file(sampled_src_data, FileUtils.handle_file_extension(src_input_file, "sample.{}".format(num_samples)))
    FileUtils.save_file(sampled_tgt_data, FileUtils.handle_file_extension(tgt_input_file, "sample.{}".format(num_samples)))


def sample_from_monolingual_data(input_file, num_samples=2000, file_type="txt"):
    data = FileUtils.load_file(input_file, file_type=file_type)
    data_size = len(data)
    ids = list(range(data_size))
    sampled_ids = random.sample(ids, k=num_samples)
    sampled_data = [data[i] for i in sampled_ids]
    FileUtils.save_file(sampled_data, FileUtils.handle_file_extension(input_file, "sample.{}".format(num_samples)))



if __name__ == "__main__":
    fire.Fire({
        "build_word_vocab": build_word_vocab,
        "sample_from_bilingual_data": sample_from_bilingual_data,
        "moses_tokenization": moses_tokenization,
        "sample_from_monolingual_data": sample_from_monolingual_data
    })
