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
import stanza
random.seed(10086)


def ner():
    stanza.download('en')


if __name__ == "__main__":
    fire.Fire({
        "ner": ner
    })
