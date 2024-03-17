from efficient_structure import MosesDataList, PhraseDataList, EncoderDataList, ChunkDataList, MosesAlignmentList
from typing import Any
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import re
import torch
from mytools.tool_utils import FileUtils, DataUtils, TorchUtils, StringUtils, logging
from transformers import AutoTokenizer
import random
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
from glob import glob
import fire


dataset_registry = {}


def register_dataset(cls):
    snake_case_name = StringUtils.camel_to_snake(cls.__name__)
    dataset_registry[snake_case_name] = cls
    return cls


def create_dataloader(ds_name: str, ds_args: dict, **named_attrs):
    ds_name = ds_name if ds_name.endswith("_dataset") else ds_name.strip() + "_dataset"
    ds = dataset_registry[ds_name](ds_args, **named_attrs)
    # if ds_args.get("batch_sampler", ""):
    #     inner_sampler = SequentialSampler(ds)
    #     batch_sampler = get_sampler_class(ds_args["batch_sampler"])(inner_sampler, ds_args['batch_size'], drop_last=False)
    #     ds_loader = DataLoader(ds, collate_fn=ds.collate_fn, batch_sampler=batch_sampler)
    # else:
    #     ds_loader = DataLoader(ds, shuffle=ds_args['shuffle'], batch_size=ds_args['batch_size'], collate_fn=ds.collate_fn)
    if ds_args.get('batch_by_token', False):
        ds_loader = DataLoader(ds, shuffle=ds_args['shuffle'], batch_size=1, collate_fn=ds.collate_fn)
    else:
        ds_loader = DataLoader(ds, shuffle=ds_args['shuffle'], batch_size=ds_args['batch_size'], collate_fn=ds.collate_fn)
    return ds_loader


def get_dataset_class(ds_name: str):
    ds_name = ds_name if ds_name.endswith("_dataset") else ds_name.strip() + "_dataset"
    return dataset_registry[ds_name] if ds_name in dataset_registry else None


@register_dataset
class RetrievalWithTokenizationDataset(Dataset):
    def __init__(self, args):
        self.args = args
        src_moses_tokenizer = DataUtils.get_moses_tokenizer(args['src_lang'])
        tgt_moses_tokenizer = DataUtils.get_moses_tokenizer(args['tgt_lang'])
        encoder_tokenizer = AutoTokenizer.from_pretrained(args['encoder_tokenizer_path'])
        max_sequence_len = encoder_tokenizer.model_max_length if args['max_sequence_len'] is None else min(encoder_tokenizer.model_max_length, args['max_sequence_len'])
        self.freq_of_stop_words = args.get('freq_of_stop_words', 10000)
        logging.info("Setting `max_sequence_len` to {}".format(max_sequence_len))
        self.encoder_tokenizer = encoder_tokenizer
        if 'data_from_pretrained' in args and args['data_from_pretrained']:
            self.load_pretrained(args['data_pretrained_ckpt'])
        else:
            max_phrase2sent_num = 6200
            src_data = FileUtils.load_file(args['src_file'])
            tgt_data = FileUtils.load_file(args['tgt_file'])
            giza_src_data, giza_tgt_data, giza_symal_data = FileUtils.load_symal(args['giza_symal_path'], return_str=True)
            moses_phrase_table_tuples = FileUtils.load_file(args['moses_phrase_table_path'])
            self.phrase_freqs = dict()

            for sp, tp, f in moses_phrase_table_tuples:
                if len(sp.split()) == 1:
                    if sp in self.phrase_freqs:
                        self.phrase_freqs[sp] += f
                    else:
                        self.phrase_freqs[sp] = f
                if len(tp.split()) == 1:
                    if tp in self.phrase_freqs:
                        self.phrase_freqs[tp] += f
                    else:
                        self.phrase_freqs[tp] = f

            self.src_tok_data, self.tgt_tok_data = [], []
            self.src_moses_data, self.tgt_moses_data = [], []
            self.alignment = [] #[[(src phrase index in sent, tgt phrase index in set)]]
            phrase2sent = defaultdict(list)
            # prepare data
            sent_id = 0
            num_tk_mismatch, num_no_al, num_over_lens, num_offset_mismatch = 0, 0, 0, 0
            self.corpus_ids = []
            logging.info("Start to prepare retrieval data")
            for corpus_id, (src, tgt) in enumerate(zip(src_data, tgt_data)):
                src_moses_tok_sent = src_moses_tokenizer.tokenize(src, escape=False, return_str=True)
                tgt_moses_tok_sent = tgt_moses_tokenizer.tokenize(tgt, escape=False, return_str=True)
                esp_src_moses_tok_sent = src_moses_tokenizer.escape_xml(src_moses_tok_sent)
                esp_tgt_moses_tok_sent = src_moses_tokenizer.escape_xml(tgt_moses_tok_sent)
                src_moses_tok = src_moses_tok_sent.split()
                tgt_moses_tok = tgt_moses_tok_sent.split()
                esp_giza_src_sent = src_moses_tokenizer.escape_xml(giza_src_data[corpus_id])
                esp_giza_tgt_sent = tgt_moses_tokenizer.escape_xml(giza_tgt_data[corpus_id])
                if not self.is_giza_moses_matching(esp_giza_src_sent, esp_src_moses_tok_sent) or not self.is_giza_moses_matching(esp_giza_tgt_sent, esp_tgt_moses_tok_sent):
                    num_tk_mismatch += 1
                    continue
                src_moses_offset = DataUtils.moses_offset(src, src_moses_tok_sent)
                tgt_moses_offset = DataUtils.moses_offset(tgt, tgt_moses_tok_sent)
                cur_moses_align = self.extract_giza_aligned_phrases(src_moses_tok, tgt_moses_tok, giza_symal_data[corpus_id], freq_of_stop_words=self.freq_of_stop_words)
                # print([(wi, w) for wi, w in enumerate(src_moses_tok)])
                # print([(wi, w) for wi, w in enumerate(tgt_moses_tok)])
                # print(giza_symal_data[corpus_id])
                # for sa, sb, ta, tb in cur_moses_align:
                #     print(src_moses_tok[sa:sb+1], tgt_moses_tok[ta:tb+1], (sa, sb), (ta, tb))
                if not cur_moses_align:
                    # logging.warning("Skip translation that has no phrase map: `{} ||| {}`".format(src, tgt))
                    num_no_al += 1
                    continue

                src_encoded = encoder_tokenizer([src], return_offsets_mapping=True, truncation=True)
                tgt_encoded = encoder_tokenizer([tgt], return_offsets_mapping=True, truncation=True)

                if len(src_encoded['input_ids'][0]) >= max_sequence_len or len(tgt_encoded['input_ids'][0]) >= max_sequence_len:
                    # logging.warning("Skip long sents: `{} ||| {}`".format(src, tgt))
                    num_over_lens += 1
                    continue

                src_moses_offset_map = DataUtils.map_tokenized_sents(src_moses_offset, src_encoded['offset_mapping'][0], skip_val_ids=[0])
                tgt_moses_offset_map = DataUtils.map_tokenized_sents(tgt_moses_offset, tgt_encoded['offset_mapping'][0], skip_val_ids=[0])

                if not self.is_valid_offset_map(src_moses_offset_map, src_moses_tok) or not self.is_valid_offset_map(tgt_moses_offset_map, tgt_moses_tok):
                    num_offset_mismatch += 1
                    continue

                for sa, sb, ta, tb in cur_moses_align:
                    # tmp_src_phrase = " ".join([src_moses_tok[k] for k in range(sa, sb)])
                    tmp_tgt_phrase = " ".join([tgt_moses_tok[k] for k in range(ta, tb+1)])
                    if len(phrase2sent[tmp_tgt_phrase]) <= max_phrase2sent_num:
                        phrase2sent[tmp_tgt_phrase].append(sent_id)
                    # if len(phrase2sent[tmp_src_phrase]) <= max_phrase2sent_num:
                    #     phrase2sent[tmp_src_phrase].append(sent_id)

                self.alignment.append(cur_moses_align)
                # (vocab id of the phrase, start idx in the RoBERTa encoded sent, end idx in the RoBERTa encoded sent)
                self.src_moses_data.append([(w, src_moses_offset_map[i][0], src_moses_offset_map[i][-1]) for i, w in enumerate(src_moses_tok) if src_moses_offset_map[i]])
                self.tgt_moses_data.append([(w, tgt_moses_offset_map[i][0], tgt_moses_offset_map[i][-1]) for i, w in enumerate(tgt_moses_tok) if tgt_moses_offset_map[i]])
                self.src_tok_data.append(src_encoded['input_ids'][0])
                self.tgt_tok_data.append(tgt_encoded['input_ids'][0])
                sent_id += 1
                self.corpus_ids.append(corpus_id)
            self.phrase2sent = dict(phrase2sent)
            self.pad_id = encoder_tokenizer.pad_token_id
            logging.info("Finished to prepare retrieval data")
            logging.info("{} sents meet the mismatch problem of moses tokenization".format(num_tk_mismatch))
            logging.info("{} sents cannot find alignment".format(num_no_al))
            logging.info("{} sents are longer than the max_length".format(num_over_lens))
            logging.info("{} sents has missmatched offset for moses and retriever tokenization".format(num_offset_mismatch))
            if 'save_pretrained_to' in args and args['save_pretrained_to']:
                self.save_pretrained(args['save_pretrained_to'])
        n = 0
        for it in self.alignment:
            n += len(it)
        logging.info("Found {} valid alignment pairs".format(n))

    def extract_giza_aligned_phrases(self, src_words, tgt_words, s2t, freq_of_stop_words=10000):

        def score_phrase(p):
            if not p.strip() or StringUtils.is_num_or_punct(p):
                # bad
                return 0
            elif self.phrase_freqs.get(p, 0) >= (2 * freq_of_stop_words):
                # slightly bad
                return 1
            elif self.phrase_freqs.get(p, 0) >= freq_of_stop_words:
                # not bad
                return 2
            else:
                # good
                return 3
            
        def add_to_list(cur_alignment, t4, force=False):
            # check empty
            for si in range(t4[0], t4[1]+1):
                p = src_words[si].strip()
                if not p:
                    return
            for ti in range(t4[2], t4[3]+1):
                p = tgt_words[ti].strip()
                if not p:
                    return
            if not cur_alignment or force:
                cur_alignment.append(t4)
            else:
                if (cur_alignment[-1][0] == t4[0] and cur_alignment[-1][1] == t4[1] and cur_alignment[-1][2] == t4[2] and cur_alignment[-1][3] <= t4[3]) or (cur_alignment[-1][2] == t4[2] and cur_alignment[-1][3] == t4[3] and cur_alignment[-1][0] == t4[0] and cur_alignment[-1][1] <= t4[1]):
                    # E.g., [i, j, m, n] is in cur_alignment, and now we find [i, j, m, n+1]
                    # Removing the [i, j, m, n] and add the [i, j, m, n+1] is better
                    popped = cur_alignment.pop(-1)
                    cur_alignment.append(t4)
                else:
                    cur_alignment.append(t4)

        i, n_s2t, cur_alignment = 0, len(s2t), []
        for a, b in s2t:
            if a >= len(src_words) or b >= len(tgt_words):
                return []
        while i < n_s2t:
            cur_sw_id, cur_tw_id = s2t[i]
            src_phrase, tgt_phrase = src_words[cur_sw_id], tgt_words[cur_tw_id]
            tmp_sw_id, tmp_tw_id, next_ptr = cur_sw_id, cur_tw_id, i+1
            if (score_phrase(src_phrase) + score_phrase(tgt_phrase)) < 5:
                i += 1
                continue
            for j in range(i+1, n_s2t):
                # check whether is one-to-many or many-to-one
                next_sw_id, next_tw_id = s2t[j]
                src_next_phrase, tgt_next_phrase = src_words[next_sw_id], tgt_words[next_tw_id]
                if next_sw_id == cur_sw_id and (next_tw_id-tmp_tw_id) == 1:
                    tmp_sw_id, tmp_tw_id = next_sw_id, next_tw_id
                    next_ptr = j+1
                elif next_tw_id == cur_tw_id and (next_sw_id-tmp_sw_id) == 1 :
                    tmp_sw_id, tmp_tw_id = next_sw_id, next_tw_id
                    next_ptr = j+1
                else:
                    break
            assert tmp_sw_id == cur_sw_id or cur_tw_id == tmp_tw_id
            add_to_list(cur_alignment, (cur_sw_id, tmp_sw_id, cur_tw_id, tmp_tw_id))
            has_post_include = False
            for j in range(next_ptr, n_s2t):
                next_sw_id, next_tw_id = s2t[j]
                src_next_phrase, tgt_next_phrase = src_words[next_sw_id], tgt_words[next_tw_id]
                if 0 <= (next_sw_id-tmp_sw_id) <= 1 and 0 <= (next_tw_id-tmp_tw_id) <= 1:
                    if next_sw_id == tmp_sw_id and (next_tw_id-tmp_tw_id) == 1 and score_phrase(src_next_phrase) > 2:
                        # one-to-many
                        has_post_include = True
                        tmp_sw_id, tmp_tw_id = next_sw_id, next_tw_id
                    elif tmp_tw_id == next_tw_id and (next_sw_id-tmp_sw_id) == 1 and not score_phrase(tgt_next_phrase) > 2:
                        # many-to-one
                        has_post_include = True
                        tmp_sw_id, tmp_tw_id = next_sw_id, next_tw_id
                    else:
                        if has_post_include:
                            add_to_list(cur_alignment, (cur_sw_id, tmp_sw_id, cur_tw_id, tmp_tw_id))
                        if (score_phrase(src_next_phrase) + score_phrase(tgt_next_phrase)) > 4:
                            add_to_list(cur_alignment, (cur_sw_id, next_sw_id, cur_tw_id, next_tw_id))
                        has_post_include = False
                        tmp_sw_id, tmp_tw_id = next_sw_id, next_tw_id
                else:
                    if has_post_include:
                        add_to_list(cur_alignment, (cur_sw_id, tmp_sw_id, cur_tw_id, tmp_tw_id))
                    break
            
            i = next_ptr
            
        return cur_alignment

    def is_giza_moses_matching(self, giza_sent, moses_sent):
        s1, s2 = giza_sent.lower(), moses_sent.lower()
        if s1 == s2 or s2.startswith(s1):
            return True
        else:
            return False

    def find_moses_phrases(self, phrase_freqs: dict, moses_tok, min_freq=1):
        # NOTE: min_freq is not determined yet
        phrase_indices = []
        n_words = len(moses_tok)
        for i in range(n_words):
            src_word_i = moses_tok[i]
            src_word_i_freq = phrase_freqs.get(src_word_i, 0)
            if StringUtils.is_number(src_word_i) or src_word_i_freq >= self.freq_of_stop_words:
                continue
            for j in range(i, n_words):
                src_word_j = moses_tok[j]
                src_word_j_freq = phrase_freqs.get(src_word_j, 0)
                if StringUtils.is_number(src_word_j) or src_word_j_freq >= self.freq_of_stop_words:
                    continue
                cur_p = " ".join(moses_tok[i:j+1])
                num = phrase_freqs.get(cur_p, 0)
                if num >= min_freq:
                    phrase_indices.append((cur_p, i, j))
        return phrase_indices

    def find_moses_alignment(self, src_moses_tok, tgt_moses_tok, topk=6):
        cur_alignment= []
        n_src_words = len(src_moses_tok)
        for src_i in range(n_src_words):
            src_word_i = src_moses_tok[src_i]
            src_word_i_freq = self.src_phrase_freqs.get(src_word_i, 0)
            if StringUtils.is_number(src_word_i) or src_word_i_freq >= self.freq_of_stop_words:
                continue
            for j in range(src_i, n_src_words):
                src_word_j = src_moses_tok[j]
                src_word_j_freq = self.src_phrase_freqs.get(src_word_j, 0)
                if StringUtils.is_number(src_word_j) or src_word_j_freq >= self.freq_of_stop_words:
                    continue
                cur_src_p = " ".join(src_moses_tok[src_i:j+1])
                if cur_src_p in self.s2t_phrase_table:
                    aligned_phases = self.s2t_phrase_table[cur_src_p][:topk]
                    for ap in aligned_phases:
                        ap_words = ap.split()
                        n_ap_w = len(ap_words)
                        if ap_words[0] in tgt_moses_tok:
                            idx = tgt_moses_tok.index(ap_words[0])
                            tmp_phrase = " ".join(tgt_moses_tok[idx:idx+n_ap_w])
                            if tmp_phrase == ap:
                                cur_alignment.append(((src_i, j+1), (idx, idx+n_ap_w)))
                else:
                    break

        return cur_alignment    
    
    def is_valid_offset_map(self, offset_map, phrase_tok):
        has_empty = False
        for it in offset_map:
            if not it:
                has_empty = True
                break
        return len(phrase_tok) == len(offset_map) and not has_empty

    def save_pretrained(self, save_path):
        logging.info("Saving dataset to {}".format(save_path))
        if not FileUtils.is_dir(save_path):
            logging.warning("`save_path={}` should be a directory, but I didn't find it".format(save_path))
            FileUtils.check_dirs(save_path)
        FileUtils.save_file(self.src_tok_data, save_path + "/src_tok_data.pt", 'pt')
        FileUtils.save_file(self.tgt_tok_data, save_path + "/tgt_tok_data.pt", 'pt')
        FileUtils.save_file(self.phrase2sent, save_path + "/phrase2sent.pt", 'pt')
        FileUtils.save_file(self.alignment, save_path + "/alignment.pt", 'pt')
        FileUtils.save_file(self.src_moses_data, save_path + "/src_moses_data.pt", 'pt')
        FileUtils.save_file(self.tgt_moses_data, save_path + "/tgt_moses_data.pt", 'pt')
        FileUtils.save_file(self.corpus_ids, save_path + "/corpus_ids.pt", 'pt')
    
    def __getitem__(self, i):
        # fake getitem
        return i
    
    def collate_fn(self, batch):
        return batch

@register_dataset
class RetrievalWithTokenizationMemoryEfficientDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.paired_data_percent = args.get("paired_data_percent", 0.7)
        self.masked_paired_data_percent = args.get("masked_paired_data_percent", 0.0)
        self.freq_of_stop_words = args.get('freq_of_stop_words', 10000)
        self.use_single_pos = args.get('use_single_pos', True)
        src_lang, tgt_lang = args['src_lang'], args['tgt_lang']
        if isinstance(src_lang, str) and isinstance(tgt_lang, str):
            src_lang = src_lang.split(",")
            tgt_lang = tgt_lang.split(",")
        assert len(src_lang) == len(tgt_lang)
        num_lang = len(src_lang)
        self.src_lang, self.tgt_lang, self.num_lang = src_lang, tgt_lang, num_lang
        src_moses_tokenizer = [DataUtils.get_moses_tokenizer(src_lang[i]) for i in range(num_lang)]
        tgt_moses_tokenizer = [DataUtils.get_moses_tokenizer(tgt_lang[i]) for i in range(num_lang)]
        encoder_tokenizer = AutoTokenizer.from_pretrained(args['encoder_tokenizer_path'])
        max_sequence_len = encoder_tokenizer.model_max_length if args['max_sequence_len'] is None else min(encoder_tokenizer.model_max_length, args['max_sequence_len'])
        logging.info("Setting `max_sequence_len` to {}".format(max_sequence_len))
        self.max_sequence_len = max_sequence_len
        self.encoder_tokenizer = encoder_tokenizer
        self.src_moses_tokenizer = src_moses_tokenizer
        self.tgt_moses_tokenizer = tgt_moses_tokenizer
        self._step = 0
        self.mask_token_id = encoder_tokenizer.mask_token_id
        self.load_pretrained(args['data_pretrained_dir_prefix'])
    
    def load_pretrained(self, data_pretrained_dir_prefix):
        logging.info("Initializing model from pre-trained")
        self.src_tok_data = EncoderDataList()
        self.tgt_tok_data = EncoderDataList()
        self.src_moses_data = MosesDataList()
        self.tgt_moses_data = MosesDataList()
        self.alignment = MosesAlignmentList()
        self.phrase2sent = []
        self.pad_id = self.encoder_tokenizer.pad_token_id
        self.langpair_start_ids = []
        self.corpus_ids = []

        for sl, tl in zip(self.src_lang, self.tgt_lang):
            save_path = data_pretrained_dir_prefix + ".{}{}".format(sl, tl)
            logging.info("Loading from pretrained: {}".format(save_path))
            if FileUtils.is_dir(save_path):
                prev_data_num = self.src_tok_data.size()
                self.langpair_start_ids.append(prev_data_num)
                
                for it in FileUtils.load_file(save_path + "/src_tok_data.pt", 'pt'):
                    self.src_tok_data.add(it)
                
                for it in FileUtils.load_file(save_path + "/tgt_tok_data.pt", 'pt'):
                    self.tgt_tok_data.add(it)
                
                for it in FileUtils.load_file(save_path + "/src_moses_data.pt", 'pt'):
                    self.src_moses_data.add(it)
                
                for it in FileUtils.load_file(save_path + "/tgt_moses_data.pt", 'pt'):
                    self.tgt_moses_data.add(it)

                for t4_vec in FileUtils.load_file(save_path + "/alignment.pt", 'pt'):
                    self.alignment.add(t4_vec)
                
                if FileUtils.exists(save_path + "/corpus_ids.pt"):
                    self.corpus_ids.extend(FileUtils.load_file(save_path + "/corpus_ids.pt", 'pt'))

                self.phrase2sent.append(FileUtils.load_file(save_path + "/phrase2sent.pt", 'pt'))
            else:
                logging.error("data checkpoint path should be a DIRECTORY!")
                raise ValueError

    def _get_phrase(self, sent_id, alignment):
        src_phrases, tgt_phrases = [], []
        src_moses_data = self.src_moses_data.index(sent_id)
        tgt_moses_data = self.tgt_moses_data.index(sent_id)
        for src_phrase_id, tgt_phrase_id in alignment:
            ss, se = src_phrase_id
            ts, te = tgt_phrase_id
            src_phrase_key = " ".join([src_moses_data[k][0] for k in range(ss, se+1)])
            tgt_phrase_key = " ".join([tgt_moses_data[k][0] for k in range(ts, te+1)])
            src_phrases.append((src_phrase_key, src_moses_data[ss][1], src_moses_data[se][-1]))
            tgt_phrases.append((tgt_phrase_key, tgt_moses_data[ts][1], tgt_moses_data[te][-1]))
        return src_phrases, tgt_phrases

    def _get_phrase_in_sent(self, sent_id, tgt_phrases):
        tgt_phrase_indices, matched_tgt_phrase_list = [], []
        tgt_moses_data = self.tgt_moses_data.index(sent_id)
        n_tgt_len = len(tgt_moses_data)
        tgt_moses_tok = [tgt_moses_data[k][0] for k in range(n_tgt_len)]
        visited_tgt_phrases = set()
        for ti, tgt_phrase in enumerate(tgt_phrases):
            if tgt_phrase[0] in visited_tgt_phrases:
                continue
            tmp_phrase_list = []
            tgt_phrase_tok = tgt_phrase[0].split()
            visited_tgt_phrases.add(tgt_phrase[0])
            n_p_len = len(tgt_phrase_tok)
            for m in range(n_tgt_len):
                if tgt_moses_tok[m] == tgt_phrase_tok[0]:
                    tmp_phrase = " ".join(tgt_moses_tok[m:m+n_p_len])
                    if tmp_phrase == tgt_phrase[0]:
                        tmp_phrase_list.append(
                            (tmp_phrase, tgt_moses_data[m][1], tgt_moses_data[m+n_p_len-1][-1])
                        )
            if tmp_phrase_list:
                matched_tgt_phrase_list.append(random.choice(tmp_phrase_list))
                tgt_phrase_indices.append(ti)
        return tgt_phrase_indices, matched_tgt_phrase_list
    
    def find_langpair_id(self, sent_id):
        langpair_id = self.num_lang - 1
        for i, start_id in enumerate(self.langpair_start_ids):
            if start_id > sent_id:
                langpair_id = i - 1
                break
        return langpair_id, self.langpair_start_ids[langpair_id]

    def find_positive(self, sent_id, use_paired_data=True, use_single_pos=True):
        alignment = self.alignment.index(sent_id)
        alignment = [((a, b), (c, d)) for a, b, c, d in alignment]
        src_sent_id, tgt_sent_id = sent_id, sent_id
        if use_single_pos:
            alignment = [random.choice(alignment)]
        src_phrases, tgt_phrases = self._get_phrase(sent_id, alignment)
        if not use_paired_data:
            al_id = np.random.choice(range(len(alignment)))
            langpair_id, langpair_start_id = self.find_langpair_id(sent_id)
            local_tgt_sent_id = random.choice(self.phrase2sent[langpair_id][tgt_phrases[al_id][0]])
            tgt_sent_id = local_tgt_sent_id + langpair_start_id
            tgt_phrase_indices, tgt_phrases = self._get_phrase_in_sent(tgt_sent_id, tgt_phrases)
            src_phrases = [src_phrases[idx] for idx in tgt_phrase_indices]

        return src_phrases, tgt_phrases, src_sent_id, tgt_sent_id

    def find_inbatch_negative(self, tgt_sent_ids, tgt_phrases, max_num=1024, is_tgt=True):
        tgt_phrase_start_ids, tgt_phrase_end_ids, tgt_phrase_batch_ids = [], [], []
        for b_id, s_id in enumerate(tgt_sent_ids):
            if is_tgt:
                phrases = set([(it[2], it[3]) for it in self.alignment.index(s_id)])
                moses_sent = self.tgt_moses_data.index(s_id)
            else:
                phrases = set([(it[0], it[1]) for it in self.alignment.index(s_id)])
                moses_sent = self.src_moses_data.index(s_id)
            for p in phrases:
                p_str = " ".join([moses_sent[w_id][0] for w_id in range(p[0], p[1]+1)])
                has_overlap = False
                for pos_phrase in tgt_phrases:
                    if p_str in pos_phrase or pos_phrase in p_str:
                        has_overlap = True
                if not has_overlap:
                    tgt_phrase_batch_ids.append(b_id)
                    tgt_phrase_start_ids.append(moses_sent[p[0]][1])
                    tgt_phrase_end_ids.append(moses_sent[p[1]][-1])
        if len(tgt_phrase_batch_ids) > max_num:
            indices = list(range(len(tgt_phrase_batch_ids)))
            np.random.shuffle(indices)
            indices = indices[:max_num]
            tgt_phrase_batch_ids = [tgt_phrase_batch_ids[i] for i in indices]
            tgt_phrase_start_ids = [tgt_phrase_start_ids[i] for i in indices]
            tgt_phrase_end_ids = [tgt_phrase_end_ids[i] for i in indices]
        return tgt_phrase_batch_ids, tgt_phrase_start_ids, tgt_phrase_end_ids

    def prepare_tkz_data(self, sent_id, is_tgt=True, max_random_span_change=3):
        if is_tgt:
            moses_sent = self.tgt_moses_data.index(sent_id)
            pos_phrases = []
            pos_phrases_indices = set([(it[2], it[3]) for it in self.alignment.index(sent_id)])
            for ms, me in pos_phrases_indices:
                phrase_str = " ".join([jt[0] for jt in moses_sent[ms:me+1]])
                pos_phrases.append((phrase_str, moses_sent[ms][1], moses_sent[me][-1]))
            sent_len = len(self.tgt_tok_data.index(sent_id))
        else:
            moses_sent = self.src_moses_data.index(sent_id)
            pos_phrases = []
            pos_phrases_indices = set([(it[0], it[1]) for it in self.alignment.index(sent_id)])
            for ms, me in pos_phrases_indices:
                phrase_str = " ".join([jt[0] for jt in moses_sent[ms:me+1]])
                pos_phrases.append((phrase_str, moses_sent[ms][1], moses_sent[me][-1]))
            sent_len = len(self.src_tok_data.index(sent_id))
        
        num_pos = max(len(pos_phrases), 1)
        max_pos_span_len = max([it[2]-it[1]+1 for it in pos_phrases])
        max_neg_span_len = np.random.randint(max_pos_span_len, max_pos_span_len + max_random_span_change)
        phrase_tkz_start_ids = [it[1] for it in pos_phrases]
        phrase_tkz_end_ids = [it[2] for it in pos_phrases]
        pos_candidates = set(it[1:] for it in pos_phrases)
        phrase_tkz_labels = [1] * num_pos
      
        # start from [1, sent_len), because sent[0] is sepcial token, i.e., [CLS]
        neg_sample_s = np.random.randint(1, sent_len-1, size=2 * num_pos)
        neg_span_len = np.random.randint(0, max_neg_span_len+1, size=2 * num_pos)
        n = 0
        for i in range(2 * num_pos):
            if n >= num_pos:
                break
            neg_phrase = (neg_sample_s[i], min(neg_sample_s[i]+neg_span_len[i], sent_len-1))
            if neg_phrase in pos_candidates:
                continue
            phrase_tkz_start_ids.append(neg_phrase[0])
            phrase_tkz_end_ids.append(neg_phrase[1])
            phrase_tkz_labels.append(0)
            n += 1
        return phrase_tkz_start_ids, phrase_tkz_end_ids, phrase_tkz_labels

    def __len__(self):
        return self.src_tok_data.size()
    
    def __getitem__(self, i):
        use_paired_data = True if np.random.uniform(0, 1) <= self.paired_data_percent else False
        src_phrases, ref_phrases, src_sent_id, tgt_sent_id = self.find_positive(i, use_paired_data=use_paired_data, use_single_pos=self.use_single_pos)
        src_tok_sent = self.src_tok_data.index(src_sent_id)
        ref_tok_sent = self.tgt_tok_data.index(tgt_sent_id)
        if use_paired_data and self.masked_paired_data_percent > 0:
            num_phrase = len(src_phrases)
            mask = np.random.rand(num_phrase) < self.masked_paired_data_percent
            for pi in range(num_phrase):
                if mask[pi]:
                    ss, se = src_phrases[pi][1], src_phrases[pi][2]
                    ts, te = ref_phrases[pi][1], ref_phrases[pi][2]
                    for wi in range(ss, se+1):
                        src_tok_sent[wi] = self.mask_token_id
                    for wi in range(ts, te+1):
                        ref_tok_sent[wi] = self.mask_token_id

        src_phrase_tkz_start_ids, src_phrase_tkz_end_ids, src_phrase_tkz_labels = self.prepare_tkz_data(src_sent_id, is_tgt=False)
        ref_phrase_tkz_start_ids, ref_phrase_tkz_end_ids, ref_phrase_tkz_labels = self.prepare_tkz_data(tgt_sent_id, is_tgt=True)

        return_dict = {
            "src_input_ids": src_tok_sent, "ref_input_ids": ref_tok_sent,
            "src_phrase_start_ids": [it[1] for it in src_phrases],
            "src_phrase_end_ids": [it[2] for it in src_phrases],
            "ref_phrase_start_ids": [it[1] for it in ref_phrases],
            "ref_phrase_end_ids": [it[2] for it in ref_phrases],
            "src_phrase_tkz_start_ids": src_phrase_tkz_start_ids,
            "src_phrase_tkz_end_ids": src_phrase_tkz_end_ids,
            "src_phrase_tkz_labels": src_phrase_tkz_labels,
            "ref_phrase_tkz_start_ids": ref_phrase_tkz_start_ids,
            "ref_phrase_tkz_end_ids": ref_phrase_tkz_end_ids,
            "ref_phrase_tkz_labels": ref_phrase_tkz_labels,
            "src_sent_id": src_sent_id,
            "tgt_sent_id": tgt_sent_id,
            "src_phrase": [it[0] for it in src_phrases],
            "ref_phrase": [it[0] for it in ref_phrases],
            "num_pos": len(src_phrases),
            "corpus_id": self.corpus_ids[i] if self.corpus_ids else i
        }
        return return_dict

    def collate_fn(self, batch):
        self._step += 1
        batch_size = len(batch)
        src_input_ids, src_phrase_batch_ids, src_phrase_start_ids, src_phrase_end_ids = [], [], [], []
        ref_input_ids, ref_phrase_batch_ids, ref_phrase_start_ids, ref_phrase_end_ids = [], [], [], []
        s2t_align_labels, t2s_align_labels = [], []
        src_phrase_tkz_batch_ids, src_phrase_tkz_start_ids, src_phrase_tkz_end_ids, src_phrase_tkz_labels = [], [], [], []
        ref_phrase_tkz_batch_ids, ref_phrase_tkz_start_ids, ref_phrase_tkz_end_ids, ref_phrase_tkz_labels = [], [], [], []
        pos_src_phrases, pos_ref_phrases = [], []
        src_sent_ids, tgt_sent_ids = [], []
        num_pos = 0
        for bid, it in enumerate(batch):
            src_input_ids.append(it['src_input_ids'])
            ref_input_ids.append(it['ref_input_ids'])
            num_pos += it['num_pos']
            src_phrase_batch_ids.extend([bid] * it['num_pos'])
            ref_phrase_batch_ids.extend([bid] * it['num_pos'])
            src_phrase_start_ids.extend(it['src_phrase_start_ids'])
            src_phrase_end_ids.extend(it['src_phrase_end_ids'])
            ref_phrase_start_ids.extend(it['ref_phrase_start_ids'])
            ref_phrase_end_ids.extend(it['ref_phrase_end_ids'])
            src_phrase_tkz_batch_ids.extend([bid] * len(it['src_phrase_tkz_start_ids']))
            src_phrase_tkz_start_ids.extend(it['src_phrase_tkz_start_ids'])
            src_phrase_tkz_end_ids.extend(it['src_phrase_tkz_end_ids'])
            src_phrase_tkz_labels.extend(it['src_phrase_tkz_labels'])
            ref_phrase_tkz_batch_ids.extend([bid] * len(it['ref_phrase_tkz_start_ids']))
            ref_phrase_tkz_start_ids.extend(it['ref_phrase_tkz_start_ids'])
            ref_phrase_tkz_end_ids.extend(it['ref_phrase_tkz_end_ids'])
            ref_phrase_tkz_labels.extend(it['ref_phrase_tkz_labels'])
            pos_src_phrases.extend(it['src_phrase'])
            pos_ref_phrases.extend(it['ref_phrase'])
            src_sent_ids.append(it['src_sent_id'])
            tgt_sent_ids.append(it['tgt_sent_id'])
        inbatch_ref_phrase_batch_ids, inbatch_ref_phrase_start_ids, inbatch_ref_phrase_end_ids = self.find_inbatch_negative(tgt_sent_ids, pos_ref_phrases, is_tgt=True)
        ref_phrase_batch_ids += inbatch_ref_phrase_batch_ids
        ref_phrase_start_ids += inbatch_ref_phrase_start_ids
        ref_phrase_end_ids += inbatch_ref_phrase_end_ids

        inbatch_src_phrase_batch_ids, inbatch_src_phrase_start_ids, inbatch_src_phrase_end_ids = self.find_inbatch_negative(src_sent_ids, pos_src_phrases, is_tgt=False)
        src_phrase_batch_ids += inbatch_src_phrase_batch_ids
        src_phrase_start_ids += inbatch_src_phrase_start_ids
        src_phrase_end_ids += inbatch_src_phrase_end_ids

        s2t_align_labels = torch.LongTensor(list(range(num_pos)))
        t2s_align_labels = torch.LongTensor(list(range(num_pos)))

        src_input_ids = [torch.LongTensor(it) for it in src_input_ids]
        ref_input_ids = [torch.LongTensor(it) for it in ref_input_ids]
        src_input_ids = pad_sequence(src_input_ids, batch_first=True, padding_value=self.pad_id)
        ref_input_ids = pad_sequence(ref_input_ids, batch_first=True, padding_value=self.pad_id)
        src_attention_mask = TorchUtils.generate_mask(src_input_ids, pad_id=self.pad_id)
        ref_attention_mask = TorchUtils.generate_mask(ref_input_ids, pad_id=self.pad_id)

        src_phrase_tkz_batch_ids, src_phrase_tkz_start_ids, src_phrase_tkz_end_ids, src_phrase_tkz_labels= TorchUtils.convert_data_to_tensor(torch.long, src_phrase_tkz_batch_ids, src_phrase_tkz_start_ids, src_phrase_tkz_end_ids, src_phrase_tkz_labels)
        src_phrase_tkz_data = torch.stack([src_phrase_tkz_batch_ids, src_phrase_tkz_start_ids, src_phrase_tkz_end_ids], dim=0)

        ref_phrase_tkz_batch_ids, ref_phrase_tkz_start_ids, ref_phrase_tkz_end_ids, ref_phrase_tkz_labels = TorchUtils.convert_data_to_tensor(torch.long, ref_phrase_tkz_batch_ids, ref_phrase_tkz_start_ids, ref_phrase_tkz_end_ids, ref_phrase_tkz_labels)
        ref_phrase_tkz_data = torch.stack([ref_phrase_tkz_batch_ids, ref_phrase_tkz_start_ids, ref_phrase_tkz_end_ids], dim=0)

        src_phrase_batch_ids, src_phrase_start_ids, src_phrase_end_ids = TorchUtils.convert_data_to_tensor(torch.long,  src_phrase_batch_ids, src_phrase_start_ids, src_phrase_end_ids)
        src_phrase_align_data = torch.stack([src_phrase_batch_ids, src_phrase_start_ids, src_phrase_end_ids], dim=0)

        ref_phrase_batch_ids, ref_phrase_start_ids, ref_phrase_end_ids = TorchUtils.convert_data_to_tensor(torch.long,  ref_phrase_batch_ids, ref_phrase_start_ids, ref_phrase_end_ids)
        ref_phrase_align_data = torch.stack([ref_phrase_batch_ids, ref_phrase_start_ids, ref_phrase_end_ids], dim=0)
        return {
            'src_input_ids': src_input_ids, "ref_input_ids": ref_input_ids,
            'src_attention_mask': src_attention_mask, "ref_attention_mask": ref_attention_mask,
            "t2s_align_labels": t2s_align_labels, 's2t_align_labels': s2t_align_labels,
            'src_phrase_align_data': src_phrase_align_data,
            'ref_phrase_align_data': ref_phrase_align_data,
            "src_phrase_tkz_data": src_phrase_tkz_data,
            "ref_phrase_tkz_data": ref_phrase_tkz_data,
            "src_phrase_tkz_labels": src_phrase_tkz_labels,
            "ref_phrase_tkz_labels": ref_phrase_tkz_labels,
        }
    
    @staticmethod
    def prepare_ngram_inference_batch(ngram_set, sents, sent_ids, encoder_tokenizer, moses_tokenizer, max_ngram_len=4, stride=1):
        aux_batch = {"phrase_batch_ids": [], "phrase_start_ids": [], "phrase_end_ids": [], "sent_ids": [], "phrase_piece": []}
        stride = max(1, max_ngram_len // 2) if stride is None else stride
        for bid, (sid, s) in enumerate(zip(sent_ids, sents)):
            moses_tok_sent = moses_tokenizer.tokenize(s, escape=False, return_str=True)
            moses_offset = DataUtils.moses_offset(s, moses_tok_sent)
            s_encoded = encoder_tokenizer([s], return_offsets_mapping=True, truncation=True)
            moses_offset_map = DataUtils.map_tokenized_sents(moses_offset, s_encoded['offset_mapping'][0], skip_val_ids=[0])
            encoded_tokens = encoder_tokenizer.convert_ids_to_tokens(s_encoded['input_ids'][0])
            num_offset = len(moses_offset_map)
            ngrams, ngram_ids = DataUtils.extract_ngrams(moses_tok_sent, max_ngram_len, return_index=True, stride=stride)
            for ng, ng_id in zip(ngrams, ngram_ids):
                ng = tuple(ng)
                if ngram_set is not None and ng in ngram_set:
                    continue
                if ng_id[0] >= num_offset or ng_id[-1] >= num_offset:
                    continue
                if not moses_offset_map[ng_id[0]] or not moses_offset_map[ng_id[-1]]:
                    continue
                phrase_start_id, phrase_end_id = moses_offset_map[ng_id[0]][0], moses_offset_map[ng_id[-1]][-1]
                ng = " ".join(encoded_tokens[phrase_start_id:phrase_end_id+1])
                aux_batch['phrase_batch_ids'].append(bid)
                aux_batch['phrase_start_ids'].append(phrase_start_id)
                aux_batch['phrase_end_ids'].append(phrase_end_id)
                aux_batch['sent_ids'].append(sid)
                aux_batch['phrase_piece'].append(ng)

        for k, v in aux_batch.items():
            if k != "phrase_piece":
                aux_batch[k] = torch.LongTensor(v)
        return aux_batch


def build_dataset(ds_name, src_lang, tgt_lang, project_root_dir, save_root_dir="", freq_of_stop_words=30000, is_study_mode=False, model_name="xlm-roberta-base"):
    # RetrievalWithTokenization
    random.seed(10086)
    np.random.seed(10086)
    prefix = "train" if not is_study_mode else "debug"
    if model_name.lower() == "labse":
        tag = "labse"
    else:
        tag = "xlmr"
    if is_study_mode:
        tag = "{}.{}".format(tag, prefix)
    if not save_root_dir:
        save_root_dir = project_root_dir
    ds_args = {
        "shuffle": True,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "max_sequence_len": 128,
        "batch_size": 8,
        "topk_as_postive": 6,
        "src_file": project_root_dir + "/wmt16_{}{}/{}.clean.{}".format(src_lang, tgt_lang, prefix, src_lang),
        "tgt_file": project_root_dir + "/wmt16_{}{}/{}.clean.{}".format(src_lang, tgt_lang, prefix, tgt_lang),
        "moses_phrase_table_path": project_root_dir + "/phrase_table/{}.clean.{}{}.moses.phtab.json".format(prefix, src_lang, tgt_lang),
        "giza_symal_path": project_root_dir + "/moses/aligned.{}.{}{}".format(prefix, src_lang, tgt_lang),
        "encoder_tokenizer_path": project_root_dir + "/huggingface/{}".format(model_name),
        "min_phrase_freq": 1,
        "min_freq_as_phrase": 1,
        'negative_num': 5,
        'freq_of_stop_words': freq_of_stop_words,
        "save_pretrained_to": save_root_dir + "/data-bin/{}.moses.withtkz.retrival.max128.fromgiza.{}{}".format(tag, src_lang, tgt_lang)
    }
    debug_dataset = get_dataset_class(ds_name)(ds_args)
    FileUtils.save_file(ds_args, save_root_dir + "/data-bin/{}.moses.withtkz.retrival.max128.fromgiza.{}{}/config.yaml".format(tag, src_lang, tgt_lang), "yaml")


def build_dataset_v2(ds_name, src_lang, tgt_lang, src_file, tgt_file, align_file, moses_phrase_table, save_dir="", freq_of_stop_words=30000, encoder_tokenizer_path="../huggingface/xlm-roberta-base"):
    # RetrievalWithTokenization
    random.seed(10086)
    np.random.seed(10086)
    ds_args = {
        "shuffle": True,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "max_sequence_len": 128,
        "batch_size": 8,
        "topk_as_postive": 6,
        "src_file": src_file,
        "tgt_file": tgt_file,
        "moses_phrase_table_path": moses_phrase_table,
        "giza_symal_path": align_file,
        "encoder_tokenizer_path": encoder_tokenizer_path,
        "min_phrase_freq": 1,
        "min_freq_as_phrase": 1,
        'negative_num': 5,
        'freq_of_stop_words': freq_of_stop_words,
        "save_pretrained_to": save_dir
    }
    debug_dataset = get_dataset_class(ds_name)(ds_args)
    FileUtils.save_file(ds_args, save_dir + "/config.yaml", "yaml")


def _test_retrieval_with_tokenization_ds_constructor():
    # RetrievalWithTokenization
    random.seed(10086)
    np.random.seed(10086)
    DEST = "./"
    ds_name = "retrieval_with_tokenization_dataset"
    ds_args = {
        "shuffle": True,
        "src_lang": "de",
        "tgt_lang": "en",
        "max_sequence_len": 128,
        "batch_size": 8,
        "topk_as_postive": 6,
        "src_phrase_tokenizer_path": DEST + "/320k.lp.mpl36.bpe.de.model.model",
        "tgt_phrase_tokenizer_path": DEST + "/320k.lp.mpl36.bpe.en.model.model",
        # "src_file": DEST + "../wmt16_data/debug.clean.de",
        # "tgt_file": DEST + "../wmt16_data/debug.clean.en",
        "src_file": DEST + "../wmt16_data/train.clean.de",
        "tgt_file": DEST + "../wmt16_data/train.clean.en",
        "moses_phrase_table_path": DEST + "../phrase_table/train.clean.moses.phtab.json",
        "giza_symal_path": DEST + "../moses/aligned.train",
        "encoder_tokenizer_path": DEST + "../huggingface/xlm-roberta-base",
        "min_phrase_freq": 1,
        "min_freq_as_phrase": 1,
        'negative_num': 5,
        'freq_of_stop_words': 30000,
        # "save_pretrained_to": DEST + "../data-bin/xlm.moses.withtkz.retrival.max128"
        "save_pretrained_to": DEST + "../data-bin/xlm.moses.withtkz.retrival.max128.fromgiza"
        # "save_pretrained_to": DEST + "../data-bin/debug.moses.withtkz.retrival.max128.fromgiza"
    }
    # debug_dataloader = create_dataloader(ds_name, ds_args)
    debug_dataset = get_dataset_class(ds_name)(ds_args)
    # for batch in debug_dataloader:
    #     continue
    # ds_args['from_pretrained'] = True
    # ds_args['pretrained_ckpt'] = ds_args["save_pretrained_to"]
    # debug_dataloader = create_dataloader(ds_name, ds_args)
    # for batch in debug_dataloader:
    #     continue



def _test_retrieval_memory_efficient_ds_constructor():
    random.seed(10086)
    np.random.seed(10086)
    DEST = "/apdcephfs/share_916081/redmondliu/intern_data/alanili/Copyisallyouneed/data/wmt16_deen/"
    ds_name = "retrieval_memory_efficient_dataset"
    ds_args = {
        "shuffle": True,
        "src_lang": "de",
        "tgt_lang": "en",
        "max_sequence_len": 128,
        "batch_size": 8,
        "topk_as_postive": 6,
        "src_phrase_tokenizer_path": DEST + "/320k.lp.mpl36.bpe.de.model.model",
        "tgt_phrase_tokenizer_path": DEST + "/320k.lp.mpl36.bpe.en.model.model",
        "src_file": DEST + "/train.clean.de",
        "tgt_file": DEST + "/train.clean.en",
        "phrase_table_path": DEST + "/train.clean.phtab.json",
        "moses_phrase_table_path": DEST + "/train.clean.moses.phtab.minfreq.3.json",
        "encoder_tokenizer_path": "/apdcephfs/share_916081/redmondliu/intern_data/alanili/huggingface/xlm-roberta-base/",
        "min_phrase_freq": 3,
        'negative_num': 5,
        "num_warmup_steps": 0,
        "max_temperature": 1.0,
        "max_train_steps": 80000,
        "save_pretrained_to": "/apdcephfs/share_916081/redmondliu/intern_data/alanili/reinvent_pbmt/data-bin/train.withmoses.retrival.max128"
    }
    # debug_dataloader = create_dataloader(ds_name, ds_args)
    ds_args['from_pretrained'] = True
    ds_args['data_pretrained_ckpt'] = ds_args["save_pretrained_to"]
    debug_dataloader = create_dataloader(ds_name, ds_args)
    for batch in debug_dataloader:
        continue

if __name__ == "__main__":
    # _test_retrieval_ds_constructor()
    # _test_retrieval_memory_efficient_ds_constructor()
    # _test_fsmt_ds_constructor()
    # _test_pbnmt_ds_constructor()
    # _test_retrieval_with_tokenization_ds_constructor()
    fire.Fire({
        "build_dataset": build_dataset,
        "build_dataset_v2": build_dataset_v2
    })
