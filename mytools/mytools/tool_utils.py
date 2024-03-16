import torch
from torch.nn.utils.rnn import pad_sequence
import json
import os
import yaml
import os.path as path
import logging
import numpy as np
import tarfile
from glob import glob
from io import BytesIO
import re
from sacremoses import MosesTokenizer, MosesDetokenizer
import math
import string
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import chain
from pympler import asizeof
import csv
import subprocess
import multiprocessing as mp
import zipfile
import urllib.parse
import random
from safetensors.torch import save_file



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)

class FileType:
    PT = "pt"
    TXT = "txt"
    JSON = "json"
    TSV = "tsv"
    XLSX = "xlsx"
    TAR = "tar"
    YAML = "yaml"
    CSV = "csv"
    NPY = "npy"
    ZIP = "zip"
    SAFETENSORS = "safetensors"
    ALL = ["pt", "txt", "json", "tsv", "xlsx", "tar", "yaml", "csv", "zip", "safetensors"]


class FileExtensionType:
    ADD = "add"
    CHANGE = "change"


# `mlg` stands for multi-lingual
LANGUAGE_LIST = ["en", "de", "zh", "vi", "fr", "cs", "fi", "ro", "ru", "tr", "ja", "mlg"]

class SystemUtils:
    @staticmethod
    def sizeof(data):
        return asizeof.asizeof(data) 
    

class MPUtils:
    @staticmethod
    def prepare_shards(data, nproc):
        if (len(data) % nproc) == 0:
            ss = len(data) // nproc
        else:
            ss = len(data) // nproc + 1
        
        shards = [data[i*ss:((i+1)*ss)] for i in range(nproc)]
        return shards

    @staticmethod
    def mp_func(target_func, args_list):
        np = len(args_list)
        processes = []
        mp.set_start_method("spawn")
        for i in range(np):
            proc = mp.Process(target=target_func, args=args_list[i])
            proc.start()
            processes.append(proc)
            logging.info("Start process {}".format(i))

        logging.info("Waiting for the finish of all processes")
        for proc in processes:
            proc.join()


class StringUtils:

    @staticmethod
    def is_number(s):
        try:
            float(s)  # for int, float, etc.
            return True
        except ValueError:
            return False

    @staticmethod
    def is_punct(s):
        return s in string.punctuation
    
    @staticmethod
    def is_num_or_punct(s):
        valid_chars = string.digits + string.punctuation
        return all(char in valid_chars for char in s)

    @staticmethod
    def is_url(s):
        try:
            result = urllib.parse.urlparse(s)
            # Check if the scheme and netloc are present
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def has_url(s):
        # Regular expression pattern for matching a broad range of URLs, including plain domain names
        for w in s.split():
            if StringUtils.is_url(w):
                return True
        return False
        
    def has_image(string):
        # Regular expression for Markdown image link
        markdown_pattern = r'!\[.*?\]\((.*?)\s*(".*?")?\)'
        html_pattern = r'<img\s+src="[^"]*"\s*(alt="[^"]*")?\s*/?>'
        return bool(re.search(html_pattern, string)) or bool(re.search(markdown_pattern, string))

    @staticmethod
    def get_digit_num(n):
        return math.ceil(math.log10(n + 1))
    
    @staticmethod
    def format_number(n, dn=None):
        if dn is None:
            dn = StringUtils.get_digit_num(n) + 3
        return "{:0{}}".format(n, dn)
    
    @staticmethod
    def camel_to_snake(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    @staticmethod
    def find_all_indices(text, substring):
        indices = []
        index = -1
        while True:
            try:
                index = text.find(substring, index + 1)
            except ValueError:
                pass
            if index == -1:
                break
            indices.append(index)
        return indices


class FileUtils:
    @staticmethod
    def load_spm(path):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(path)
        return sp

    @staticmethod
    def split_and_convert_pdf_to_jpeg(fname_wo_ext):
        assert not fname_wo_ext.endswith("pdf")
        log = subprocess.run(
            ["pdftoppm", "-jpeg", "{}.pdf".format(fname_wo_ext), fname_wo_ext],
            capture_output=True)
        return log

    @staticmethod
    def exists(file_path):
        return path.exists(file_path)
    
    @staticmethod
    def listdir(dir_path, return_full_name=False, only_folder=False, only_file=False):
        files = []
        for entry in os.listdir(dir_path):
            full_path = os.path.join(dir_path, entry)
            is_folder = os.path.isdir(full_path)
            if only_folder and is_folder:
                files.append(entry if not return_full_name else full_path)
            elif only_file and not is_folder:
                files.append(entry if not return_full_name else full_path)
            else:
                files.append(entry if not return_full_name else full_path)
        return files

    
    @staticmethod
    def rename(source_fname, target_fname):
        if os.path.exists(source_fname):
            # Rename the file
            os.rename(source_fname, target_fname)
            logging.info("File renamed from {} to {}".format(source_fname, target_fname))
        else:
            logging.info("The file {} does not exist".format(source_fname))

    @staticmethod
    def is_dir(file_path):
        return path.isdir(file_path)

    @staticmethod
    def get_last_path(path):
        if not path:
            return path
        parent, last_path = os.path.split(path)
        if last_path:
            return last_path
        else:
            return FileUtils.get_last_path(parent)

    @staticmethod
    def load_giza_a3_final(giza_out_dir):
        file_names = sorted(list(glob(giza_out_dir + "/alignment.A3.final.part*")))
        align_data = dict()
        for fn in file_names:
            data = FileUtils.load_file(file_names, 'txt')
            data_size = len(data)
            assert (data_size % 3) == 0
            for i in range(0, data_size, 3):
                sent_id, align_ids = DataUtils.parse_giza(data[i:i+3])
                align_data[sent_id] = align_ids
        return align_data
    
    @staticmethod
    def load_symal(symal_path, return_str=False):
        data = FileUtils.load_file(symal_path, 'txt')
        src_lens, tgt_lens, aligns = [], [], []
        src_data, tgt_data = [], []
        for line in data:
            src, tgt, align = DataUtils.parse_symal(line)
            src_data.append(src)
            tgt_data.append(tgt)
            src_lens.append(len(src.split()))
            tgt_lens.append(len(tgt.split()))
            aligns.append(align)
        if not return_str:
            return src_lens, tgt_lens, aligns
        else:
            return src_data, tgt_data, aligns


    @staticmethod
    def get_dir(path):
        return os.path.dirname(path)

    @staticmethod
    def check_dirs(dir_path):
        if path.exists(dir_path):
            logging.info("{} already exists".format(dir_path))
        else:
            logging.info("Making new directory {}".format(dir_path))
            os.makedirs(dir_path)

    @staticmethod
    def check_basename(fpath):
        bname = os.path.basename(fpath)
        parts = bname.split(".")
        if len(parts) <= 1:
            return bname
        elif parts[-1] in LANGUAGE_LIST or parts[-1] in FileType.ALL:
            return ".".join(parts[:-1])
        else:
            return bname

    @staticmethod
    def check_file_type(fpath):
        parts = fpath.split(".")
        ext = ""
        if parts:
            ext = parts[-1]
        return ext
    
    @staticmethod
    def load_shards(prefix, suffix, n_shards, flat_data=True):
        data = []
        for i in range(n_shards):
            cur_data = FileUtils.load_file("{}.{}.{}".format(prefix, i, suffix))
            if isinstance(cur_data, list) and flat_data:
                data += cur_data
            else:
                data.append(cur_data)
        return data

    @staticmethod
    def data_iterator(file_pattern, file_type=None, shard_size=0):
        fpath_list = sorted(list(glob(file_pattern)))
        if fpath_list:
            logging.info("Files will be loaded in the following order:\n{}".format(
                "\n".join(fpath_list)
            ))
        else:
            logging.warning("No file found given this pattern: {}".format(file_pattern))
        shard_data = []
        for fpath in fpath_list:
            logging.info("Start to process {}".format(fpath))
            loaded_data = FileUtils.load_file(fpath, file_type)
            if shard_size > 0:
                shard_data += loaded_data
                while len(shard_data) >= shard_size:
                    yield shard_data[:shard_size]
                    shard_data = shard_data[shard_size:]
            else:
                for d in loaded_data:
                    yield d
        if shard_size > 0 and shard_data:
            yield shard_data
        
    @staticmethod
    def load_from_disk(fpath, file_tyle=None):
        return FileUtils.load_file(fpath, file_tyle)
    
    @staticmethod
    def file_names_in_zip(fpath):
        with zipfile.ZipFile(fpath, 'r') as fzip:
            return fzip.namelist()
        
    @staticmethod
    def load_zip(zip_file_path, file_names=None):
        file_content = {}
        with zipfile.ZipFile(zip_file_path, 'r') as fzip:
            all_files = fzip.namelist()
            if file_names is None:
                file_names = all_files
            all_files = set(all_files)
            for fname in file_names:
                if fname in all_files:
                    with fzip.open(fname) as fin:
                        file_content[fname] = fin.read().decode()
        return file_content

    @staticmethod
    def load_file(fpath, file_type=None):
        if file_type is None:
            file_type = FileUtils.check_file_type(fpath)
        if file_type == FileType.TXT:
            data = []
            with open(fpath, 'r', errors='ignore') as fin:
                for line in fin:
                    data.append(line.strip())
        elif file_type == FileType.PT:
            data = torch.load(fpath)
        elif file_type == FileType.JSON:
            with open(fpath, 'r') as fin:
                data = json.load(fin)
        elif file_type == FileType.TSV:
            data = []
            with open(fpath, 'r') as fin:
                for line in fin:
                    data.append(line.strip().split("\t"))
        elif file_type == FileType.XLSX:
            import openpyxl
            data = []
            workbook = openpyxl.load_workbook(fpath)
            worksheet = workbook.active
            for row in worksheet.iter_rows(values_only=True):
                data.append(list(row))
        elif file_type == FileType.CSV:
            data = dict()
            with open(fpath) as fin:
                reader = csv.DictReader(fin)
                col_names = reader.fieldnames
                data = [col_names]
                for row in reader:
                    data.append([row[cn] for cn in col_names])
        elif file_type == FileType.TAR:
            from PIL import Image
            # NOTE: this is only for the caption & img data
            data = []
            with tarfile.open(fpath, "r") as tar:
                txt_files = sorted([f.name for f in tar.getmembers() if f.name.endswith('.txt')])
                jpg_files = [FileUtils.handle_file_extension(it, 'jpg', 'change', True) for it in txt_files]
                for tf, jf in zip(txt_files, jpg_files):
                    txt_obj = tar.extractfile(tf)
                    jpg_obj = tar.extractfile(jf)
                    txt_data = txt_obj.read().decode("utf-8")
                    jpg_data = Image.open(BytesIO(jpg_obj.read()))
                    data.append((txt_data, jpg_data))
        elif file_type == FileType.YAML:
            with open(fpath, 'r') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
        elif file_type == FileType.NPY:
            data = np.load(fpath)
        elif file_type == FileType.ZIP:
            data = FileUtils.load_zip_files(fpath)
        elif file_type == FileType.SAFETENSORS:
            from safetensors import safe_open
            data = {}
            with safe_open(fpath, framework="pt") as f:
                for k in f.keys():
                    data[k] = f.get_tensor(k)
        else:
            logging.warning("Unknown loading file type: {}".format(file_type))
            if file_type in LANGUAGE_LIST:
                data = []
                logging.info("Treat file with language suffix {} by txt".format(file_type))
                with open(fpath, 'r') as fin:
                    for line in fin:
                        data.append(line.strip())
            else:
                data = torch.load(fpath)
        logging.info("Loaded file from {}".format(fpath))
        return data

    @staticmethod
    def save_file(data, fpath, file_type=None):
        FileUtils.save_to_disk(data, fpath, file_type)

    @staticmethod
    def save_to_disk(data, fpath, file_type=None):
        if file_type is None:
            file_type = FileUtils.check_file_type(fpath)

        if file_type == FileType.TXT:
            with open(fpath, 'w') as fout:
                for line in data:
                    fout.write("{}\n".format(line.strip()))
        elif file_type == FileType.PT:
            torch.save(data, fpath)
        elif file_type == FileType.JSON:
            with open(fpath, 'w') as fout:
                json.dump(data, fout, indent="\t")
        elif file_type == FileType.TSV:
            with open(fpath, 'w') as fout:
                for it in data:
                    fout.write("{}\n".format("\t".join(it).strip()))
        elif file_type == FileType.YAML:
            with open(fpath, 'w') as fout:
                yaml.dump(data, fout)
        elif file_type == FileType.CSV:
            with open(fpath, 'w') as fout:
                writer = csv.DictWriter(fout)
                for row in data:
                    writer.writerow(row)
        elif file_type == FileType.NPY:
            np.save(fpath, data)
        elif file_type == FileType.SAFETENSORS:
            from safetensors.torch import save_file
            if not isinstance(data, dict):
                logging.warning("Safetensors only accept dict, but I got {}".format(type(data)))
            save_file(data, path)
        else:
            logging.warning("Unknown saving file type: {}".format(file_type))
            if file_type in LANGUAGE_LIST:
                logging.info("Treat file with language suffix {} by txt".format(file_type))
                with open(fpath, 'w') as fout:
                    for line in data:
                        fout.write("{}\n".format(line.strip()))
            else:
                torch.save(data, fpath)
        logging.info("Save file to {}".format(fpath))
    
    @staticmethod
    def handle_file_extension(file_path, new_extension, type=FileExtensionType.ADD, only_return_basename=False):
        from pathlib import Path
        # Ensure the new extension starts with a dot
        if not new_extension.startswith("."):
            new_extension = f".{new_extension}"
        file = Path(file_path)
        if type == FileExtensionType.CHANGE:
            new_file_name = f"{file.parent}/{file.stem}{new_extension}"
        elif type == FileExtensionType.ADD:
            new_file_name = f"{file.parent}/{file.stem}{new_extension}{file.suffix}"
        if only_return_basename:
            return os.path.basename(new_file_name)
        else:
            return new_file_name


class DataUtils:
    @staticmethod
    def unique_item_indices(data):
        indices = []
        unique_set = set()
        for i, it in enumerate(data):
            if it not in unique_set:
                indices.append(i)
                unique_set.add(it)
        return indices

    @staticmethod
    def chain(data):
        return list(chain(*data))

    @staticmethod
    def tuple2dict(data, key_idx=0, val_idx=1, criteria_fn=None):
        """
        criteria_fn: use the tuple if criteria_fn returns true
        """
        dict_data = defaultdict(list)
        if criteria_fn is None:
            for it in data:
                dict_data[it[key_idx]].append(it[val_idx])
        else:
            for it in data:
                if criteria_fn(it):
                    dict_data[it[key_idx]].append(it[val_idx])            
        return dict(dict_data)

    @staticmethod
    def get_moses_tokenizer(lang):
        return MosesTokenizer(lang=lang)

    @staticmethod
    def get_moses_detokenizer(lang):
        return MosesDetokenizer(lang=lang)

    @staticmethod
    def moses_offset(orig_sent: str, moses_tok_sent: str):
        for escape_char, orig_char in MosesDetokenizer.MOSES_UNESCAPE_XML_REGEXES:
            moses_tok_sent = re.sub(escape_char, orig_char, moses_tok_sent)
        indexes = []
        i = 0
        for word in moses_tok_sent.split():
            index = orig_sent.find(word, i)
            if index < 0:
                return []
            indexes.append((index, index+len(word)))
            i = index + len(word)
        return indexes
    
    @staticmethod
    def parse_symal(line):
        al_info = line.strip().split(" {##} ")
        if al_info == "ALIGN_ERR":
            return "", "", 0
        else:
            align_pairs = []
            for it in al_info[-1].split():
                if "-" in it:
                    a, b = it.split("-")
                else:
                    try:
                        a, b = it.split("p")
                    except Exception:
                        import pdb; pdb.set_trace()
                align_pairs.append((int(a), int(b)))
            align_pairs = sorted(align_pairs, key=lambda x: x[0])
            return al_info[0], al_info[1], align_pairs

    @staticmethod
    def parse_giza(giza_output):
        if isinstance(giza_output, str):
            giza_output = giza_output.strip().split('\n')
        sentence_id = int(re.search(r'\((\d+)\)', giza_output[0]).group(1))
        aligned_ids = []
        matches = re.findall(r'\{([^}]*)\}', giza_output[2])
        for m in matches:
            # Sicne word id starts from 1 in GIZA++, so we substract 1 here
            aligned_ids.append([int(it)-1 for it in m.split()])
        return sentence_id, aligned_ids

    @staticmethod
    def remove_duplicate_text(data):
        unique_lines = dict()
        for it in data:
            unique_lines[it.lower()] = it
        return list(unique_lines.values())

    @staticmethod
    def sp_encode(sp, sents, return_offsets_mapping=False, out_type=str, enable_sampling=False):
        pieces, offset = [], []
        proto = sp.encode(sents, out_type='immutable_proto', enable_sampling=enable_sampling, alpha=0.1)
        if isinstance(sents, str):
            for it in proto.pieces:
                if out_type == str:
                    pieces.append(it.piece)
                elif out_type == int:
                    pieces.append(it.id)
                else:
                    raise ValueError
                offset.append((it.begin, it.end))
        elif isinstance(sents, list):
            for cur_proto in proto:
                cur_pieces, cur_offset  = [], []
                for it in cur_proto.pieces:
                    if out_type == str:
                        cur_pieces.append(it.piece)
                    elif out_type == int:
                        pieces.append(it.id)
                    else:
                        raise ValueError
                    cur_offset.append((it.begin, it.end))
                pieces.append(cur_pieces)
                offset.append(cur_offset)
        else:
            raise NotImplementedError
        if return_offsets_mapping:
            return pieces, offset
        else:
            return pieces
        # else:
        #     if out_type == str:
        #         return sp.encode(sents, out_type=str)
        #     elif out_type == int:
        #         return sp.encode(sents, out_type=int)
        #     else:
        #         raise ValueError

    @staticmethod
    def has_overlap(a_span, b_span):
        a_s, a_e = a_span
        b_s, b_e  = b_span
        return a_s <= b_s <= a_e or b_s <= a_s <= b_e

    @staticmethod
    def map_tokenized_sents(key_offset, val_offset, skip_val_ids=[]):
        ret = [[] for _ in range(len(key_offset))]
        val_ptr, val_end_idx = 0, len(val_offset)
        for i, (key_s, key_e) in enumerate(key_offset):
            while val_ptr < val_end_idx:
                val_s, val_e = val_offset[val_ptr]
                if val_s <= key_s <= val_e or key_s <= val_s <= key_e:
                    if val_ptr not in skip_val_ids:
                        ret[i].append(val_ptr)
                    if key_e <= val_e:
                        if key_e == val_e:
                            val_ptr += 1
                        break
                    else:
                        val_ptr += 1
                else:
                    # logging.info("Probabily ERROR: \nOffset 1: {}\nOffset 2: {}".format(val_offset, key_offset))
                    # import pdb; pdb.set_trace()
                    break
        return ret

    @staticmethod
    def load_spm(path):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(path)
        return sp

    @staticmethod
    def extract_ngrams(sent, n, return_index=False, stride=1):
        if isinstance(sent, str):
            sent = sent.split()
        ngrams = [tuple(sent[i:i + n]) for i in range(0, len(sent) - n + 1, stride)]
        if not return_index:
            return ngrams
        else:
            ids = list(range(len(sent)))
            ngram_ids = [tuple(ids[i:i + n]) for i in range(0, len(ids) - n + 1, stride)]
            return ngrams, ngram_ids
    
    @staticmethod
    def num_and_punc_rate(sent):
        num_and_punc_set = set(string.digits + string.punctuation)
        n = 0
        for c in sent:
            if c in num_and_punc_set:
                n += 1
        return n / len(sent)


class HFUtils:

    @staticmethod
    def avg_hidden_states(h, mask=None, dim=1):
        if mask is not None:
            h = h * mask.unsqueeze(-1)
            h_len = mask.sum(dim=-1, keepdim=True)
            h = h.sum(dim=dim) / h_len
        else:
            h = h.mean(dim=dim)
        return h

    @staticmethod
    def encode(model, tokenizer, batch, device, mask_padding=True, is_mm=False):
        mask = None
        with torch.no_grad():
            if not is_mm:
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            else:
                inputs = tokenizer(text=batch['text'], images=batch['images'], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if mask_padding:
                mask = inputs["attention_mask"]
            outputs = model(**inputs, output_hidden_states=True)
        return outputs, mask

    @staticmethod
    def load_extracted_repr(index_file, repr_file, normalize=True, log=False):
        idx_data = torch.load(index_file)
        repr_data = torch.load(repr_file)
        assert isinstance(idx_data[0], int) and isinstance(repr_data[0], torch.Tensor)
        ids_array = np.array(idx_data)
        if isinstance(repr_data, list):
            xb_array = torch.stack([it.view(-1) for it in repr_data], dim=0).float().cpu().numpy()
        elif isinstance(repr_data, torch.Tensor):
            xb_array = repr_data.cpu().numpy()
        elif isinstance(repr_data, numpy.ndarray):
            xb_array = repr_data
        else:
            raise TypeError("Unkown type for the representation data: {}".format(type(repr_data[0])))
        assert len(xb_array.shape) == 2
        if normalize:
            xb_array = xb_array / np.expand_dims(np.linalg.norm(xb_array, axis=-1), axis=-1)
        if log:
            xb_array = np.log(xb_array)
        return xb_array, ids_array


class FaissUtils:
    @staticmethod
    def to_single_gpu(index, gpuid=0, useFloat16=False):
        import faiss
        res = faiss.StandardGpuResources()
        if useFloat16:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False
        else:
            co = None
        return faiss.index_cpu_to_gpu(res, gpuid, index, co)

    @staticmethod
    def search(xq, index, batch_size, topk):
        hist_I = None
        hist_D = None
        batch_num = xq.shape[0] // batch_size
        if (xq.shape[0] % batch_size) != 0:
            batch_num += 1

        for i in range(batch_num):
            s, e = i * batch_size, (i+1) * batch_size
            D, I = index.search(xq[s:e], topk)
            if hist_I is None:
                hist_I = I
                hist_D = D
            else:
                hist_I = np.concatenate((hist_I, I), axis=0)
                hist_D = np.concatenate((hist_D, D), axis=0)

        return hist_D, hist_I
    

class TorchUtils:

    @staticmethod
    def set_seed(seed: int):
        """
        Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``.

        Args:
            seed (:obj:`int`): The seed to set.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def generate_mask(ids, pad_id=0):
        '''generate the mask matrix of the ids, default padding token idx is 0'''
        mask = torch.ones_like(ids)
        mask[ids == pad_id] = 0.
        return mask
    
    @staticmethod
    def batchfy(sequences, pad_id=0):
        device = sequences[0].device
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_id)
        attention_mask = TorchUtils.generate_mask(padded_sequences, pad_id).to(device)
        return padded_sequences, attention_mask


    @staticmethod
    def convert_data_to_tensor(dtype, *args):
        return [torch.tensor(d, dtype=dtype) for d in args]
    
    @staticmethod
    def move_to_device(data, device):
        '''map the tensor on cuda device'''
        if isinstance(data, Mapping):
            ret_data = dict()
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    ret_data[k] = v.to(device)
                else:
                    ret_data[k] = torch.tensor(v, device=device)
        elif isinstance(data, Sequence) and not isinstance(data, str):
            ret_data = []
            for v in data:
                if isinstance(v, torch.Tensor):
                    ret_data.append(v.to(device))
                else:
                    ret_data.append(torch.tensor(v, device=device))
        else:
            logging.error("The data type `{}` is not supported".format(type(data)))
            raise RuntimeError
        return ret_data