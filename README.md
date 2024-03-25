## 1. Introduction

The code for our paper Cross-lingual Contextualized Phase Retrieval.

## 2. TODO List

- [ ] Unify the python environment
- [ ] Release the human annotated data for retrieval
- [x] Release the code for training
- [ ] Release the pre-trained model
- [x] Release the code for retrieval inference
- [ ] Release the code for MT inference

## 3. Environment

> TODO: Unify the python environment

1. **[Preparing Training Data]** GIZA++ requires python2.7
2. **[Training Model]** Our project requires python3.9 + transformers==4.27.1
3. **[SFT LLM for MT]** Platypus requires transformers>=4.31.0

Below is the explanation of each folder:

1. `mytools`: This is a library containing commonly used functions. 
2. `mgiza`: The Code of GIZA++ for preparing the training data of CCPR.
3. `code`: The main code for our project.
4. `Platypus`: The code for fine-tuning the LLM for MT 

**!!!Please install those libraries according to their README!!!**

## 4. Download

### 4.1 HF Model

Please ensure the `mytools` library is installed.

```bash
python ./mytools/mytools/hf_tools.py download_hf_model "sentence-transformers/LaBSE" ./huggingface/LaBSE
python ./mytools/mytools/hf_tools.py download_hf_model "FacebookAI/xlm-roberta-base" ./huggingface/xlm-roberta-base
```

### 4.2 HF Dataset

```bash
for L1 in "de" "cs" "fi" "ru" "ro" "tr"
do
    python ./mytools/mytools/hf_tools.py download_wmt --ds_name wmt16 --lang_pair ${L1}-en --save_path ./wmt16_${L1}en
done
```

### 4.3 Human-annotated Word Alignment

Please download the pre-processed data-bin from this link (TODO), and put it to `./data-bin`.

If you want to pre-process the data of human-annotated word alignment by youself, please download the raw data as follows:

|        |                                     URL                                     | Re-name           |
|--------|:---------------------------------------------------------------------------:|-------------------|
| De->En | https://www-i6.informatik.rwth-aachen.de/goldAlignment/                     | ./align_data/DeEn |
| Cs->En | https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1804             | ./align_data/CzEn |
| Ro->En | http://web.eecs.umich.edu/~mihalcea/wpt05/data/Romanian-English.test.tar.gz | ./align_data/RoEn |


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

If you don't want to use the pre-trained model, please see the section below to train your own model

**Step-2**: build index


**Step-3**: retrieval


**Step-4**: fine-tune LLM for translation

**Step-5**: decode


**Step-6**: evaluation




### 5.3: Model Training
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
