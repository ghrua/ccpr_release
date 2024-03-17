## TODO List

- [ ] Unify the python environment
- [ ] Release the human annotated data for retrieval
- [ ] Release the code for training
- [ ] Release the pre-trained model
- [ ] Release the code for inference

## Environment

> TODO: Unify the python environment

1. **[Preparing Training Data]** GIZA++ requires python2.7
2. **[Training Model]** Our project requires python3.9 + transformers==4.27.1
3. **[SFT LLM for MT]** Platypus requires transformers>=4.31.0

Below is the explanation of each folder:

1. `mytools`: This is a library containing commonly used functions. **Please install this library according to its README first.**
2. `mgiza`: The Code of GIZA++ for preparing the training data of CCPR.
3. `code`: The main code for our project.
4. `Platypus`: The code for fine-tuning the LLM for MT 

## Download

### HF Model

Please ensure the `mytools` library is installed.

```bash
python ./mytools/mytools/hf_tools.py download_hf_model "sentence-transformers/LaBSE" ./huggingface/LaBSE
python ./mytools/mytools/hf_tools.py download_hf_model "FacebookAI/xlm-roberta-base" ./huggingface/xlm-roberta-base
```

### HF Dataset

```bash
for L1 in "de" "cs" "fi" "ru" "ro" "tr"
do
    python ./mytools/mytools/hf_tools.py download_wmt --ds_name wmt16 --lang_pair ${L1}-en --save_path ./wmt16_${L1}en
done
```

### Human-annotated Word Alignment

Please download the pre-processed data from [this link](xxx), and put it to `./datastore`.

If you want to pre-process the data of human-annotated word alignment by youself, please download the raw data as follows:

|        |                                     URL                                     | Re-name           |
|--------|:---------------------------------------------------------------------------:|-------------------|
| De->En | https://www-i6.informatik.rwth-aachen.de/goldAlignment/                     | ./align_data/DeEn |
| Cs->En | https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1804             | ./align_data/CzEn |
| Ro->En | http://web.eecs.umich.edu/~mihalcea/wpt05/data/Romanian-English.test.tar.gz | ./align_data/RoEn |


## Ussage

### Inference: Retrieval

```bash
cd code
bash eval_retriever.sh
```

### Inference: MT


### Preaprocess Training Data

Run the following code to automatically induce the cross-lingual phrase pairs from the parallel data.

```bash
cd code
bash preprocess_data.sh
```


### Model Training

```bash
cd code
bash train_retriever_labse_multilingual.sh
```
