## 1. Introduction

The code for our paper Cross-lingual Contextualized Phase Retrieval. 

> NOTE: Since this project contains many pipelines and each part is finished seperately during this long-term project, I have not test the whole project from scratch again, which is one thing in my TODO list. However, I think the code and scrips are helpful for people who are curious about how we implement our method. Please feel free to ask any questions about this project. My email address: li.huayang.lh6 [at] is.naist.jp

## 2. TODO List

- [ ] Unify the python environment
- [ ] Test those scripts one more time
- [x] Release the human annotated data for retrieval
- [x] Release the code for training
- [x] Release the pre-trained model
- [x] Release the code for retrieval inference
- [x] Release the code for MT inference

## 3. Environment

> TODO: Unify the python environment

1. **[Preparing Training Data]** GIZA++ requires python2.7
2. **[Training Model]** Our project requires python3.9 + transformers==4.27.1
3. **[SFT LLM for MT]** Platypus requires transformers>=4.31.0

Below is the short explanation about four critical folders:

1. `mytools`: This is a library containing commonly used functions in this project, such as reading and saving files. 
2. `mgiza`: The code of GIZA++ for automatically inducing word alignment information from parallel data, which is important for collecting training data for CCPR.
3. `code`: The main code for our project, including the code for model, dataloader, indexing, searching, etc.
4. `Platypus`: The code for LLM-based translator. In our paper, we use the CCPR model to augment the LLM-based translator by integrating the retrieved information to the LLM prompt.

**!!!Please install those libraries according to their README files!!!**

## 4. Download

### 4.1 HF Model

Please ensure the `mytools` library is installed.

```bash
python ./mytools/mytools/hf_tools.py download_hf_model "sentence-transformers/LaBSE" ./huggingface/LaBSE
python ./mytools/mytools/hf_tools.py download_hf_model "FacebookAI/xlm-roberta-base" ./huggingface/xlm-roberta-base
```

Please also make sure you have the checkpoint of Llama-2-7B, which will be used for the MT task.

### 4.2 HF Dataset

```bash
for L1 in "de" "cs" "fi" "ru" "ro" "tr"
do
    python ./mytools/mytools/hf_tools.py download_wmt --ds_name wmt16 --lang_pair ${L1}-en --save_path ./wmt16_${L1}en
done
```

### 4.3 Human-annotated Word Alignment (for Retrieval Evaluation)

Please download the pre-processed data-bin from [this link](https://drive.google.com/file/d/1xCa8xrQrx-O8lxM2Y0WNadUpqiLBPJ-w/view?usp=share_link), and put it to the root directory of this project, i.e., this folder.

If you want to pre-process the data of human-annotated word alignment by youself, please download the raw data as follows:

|        |                                     URL                                     | Re-name           |
|--------|:---------------------------------------------------------------------------:|-------------------|
| De->En | https://www-i6.informatik.rwth-aachen.de/goldAlignment/                     | ./align_data/DeEn |
| Cs->En | https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1804             | ./align_data/CzEn |
| Ro->En | http://web.eecs.umich.edu/~mihalcea/wpt05/data/Romanian-English.test.tar.gz | ./align_data/RoEn |


### 4.4 Newscrawl Monolingual Data (for MT Evaluation)


```bash
# make sure you are under the root directory of the project
mkdir -p newscrawl
cd newscrawl
YY=16 # an example
LANG=tr # an example
wget https://data.statmt.org/news-crawl/${LANG}/news.20${YY}.${LANG}.shuffled.deduped.gz
gzip -d news.20${YY}.${LANG}.shuffled.deduped.gz
```
where `YY` is the number of year, e.g., `16`, and `LANG` is the language of the data, e.g, `tr`.

## 5. Ussage

### 5.1 Inference: Retrieval

Before runing the following script, please remember to complete some configs in the script, e.g., path to python, and also make sure that you have installed the required libraries.

```bash
cd code
bash eval_retriever.sh
```
If you want to pre-process your own data for retrieval, please check un-comment the code for data processing in `eval_retriever.sh`.

### 5.2 Inference: MT

> TODO

**Step-1**: Model training 

Please save unzip the [pre-trained retriever](https://drive.google.com/file/d/1baMaqob6Q09kESNwG-7wtNyHhf7c0RaS/view?usp=share_link) and save the `ckpts` folder to the root path of this project (this folder). If you don't want to use the pre-trained model, please see the Section 5.3 to train your own model. 

**Step-2**: Data Processing & Indexing & searching

Please make sure you have downloaded the newscrawl monolingual data and install the required libraries.

```bash
cd code
bash index_and_search.sh
```

**Step-4**: Fine-tune LLM for translation

Please save unzip the [pre-trained LLM-based translator](https://drive.google.com/file/d/17JONPq1J7QfxR3C83b5l9mw-vZXDGmAl/view?usp=share_link) and save the `ckpts` folder to the root path of this project (this folder). If you don't want to use the pre-trained model, prepare training data and train it by yourself following the README of Platypus. 

```bash
cd Platypus
bash fine-tuning.sh
```

**Step-5**: Decoding & Reprting Score

You can use the folllowing data to prepare the prompts. Note you can prepare the training data using the following script, too.
```bash
cd Platypus
bash prepare_alpaca_data_phrase_test_enxx.sh
```

Then, run the script for decoding and evaluation. The score for the method will be printed.

```bash
cd Platypus
bash inference_test.sh
```


### 5.3: CCPR Training
Please check whether you need set project configs, e.g., project path, for each script.

**Step-1**: get the word alignment information from parallel data using GIZA++
```bash
cd mgiza/mgizapp/build
bash install.sh
bash run.sh
```

**Step-2**: run the following code to automatically induce the cross-lingual phrase pairs from the parallel data.
```bash
cd code
bash preprocess_data.sh
```

**Step-3**: train model

```bash
cd code
bash train_retriever_labse_multilingual.sh
```
