########################################################
# Hyper-prameters
########################################################
PYTHON_PATH=/path/to/python/

########################################################
# Prepare alignment data
########################################################

$PYTHON_PATH/python eval_retriever.py prepare_deen_annotation_data
$PYTHON_PATH/python eval_retriever.py prepare_czen_annotation_data
$PYTHON_PATH/python eval_retriever.py prepare_roen_annotation_data


ENCODER_TOKENIZER_PATH=../huggingface/xlm-roberta-base
ENCODER_TOKENIZER_PATH=../huggingface/bert-base-multilingual-cased
FREQ_OF_STOP_WORDS=30000
MODEL_TYPE=mbert

L1=de
L2=en
python dataloader.py build_dataset_v2 \
--ds_name "retrieval_with_tokenization_dataset" \
--src_lang $L1 \
--tgt_lang $L2 \
--src_file ../wmt16_deen/eval.clean.$L1 \
--tgt_file ../wmt16_deen/eval.clean.$L2 \
--align_file ../moses/aligned.eval.${L1}${L2} \
--encoder_tokenizer_path $ENCODER_TOKENIZER_PATH \
--moses_phrase_table ..//phrase_table/train.clean.deen.moses.phtab.json \
--freq_of_stop_words  $FREQ_OF_STOP_WORDS \
--save_dir ../data-bin/$MODEL_TYPE.eval.retriever.train.${L1}${L2}

EVAL_L1="de"
EVAL_L2="en"
python dataloader.py build_dataset_v2 \
--ds_name "retrieval_with_tokenization_dataset" \
--src_lang $EVAL_L1 \
--tgt_lang $EVAL_L2 \
--src_file ../align_data/DeEn/test.clean.$EVAL_L1 \
--tgt_file ../align_data/DeEn/test.clean.$EVAL_L2 \
--align_file ../align_data/DeEn/aligned.test.${EVAL_L1}${EVAL_L2} \
--encoder_tokenizer_path $ENCODER_TOKENIZER_PATH \
--moses_phrase_table ..//phrase_table/train.clean.deen.moses.phtab.json \
--freq_of_stop_words  $FREQ_OF_STOP_WORDS \
--save_dir ../data-bin/$MODEL_TYPE.eval.retriever.test.${EVAL_L1}${EVAL_L2}


EVAL_L1="cs"
EVAL_L2="en"
python dataloader.py build_dataset_v2 \
--ds_name "retrieval_with_tokenization_dataset" \
--src_lang $EVAL_L1 \
--tgt_lang $EVAL_L2 \
--src_file ../align_data/CzEn/test.clean.$EVAL_L1 \
--tgt_file ../align_data/CzEn/test.clean.$EVAL_L2 \
--align_file ../align_data/CzEn/aligned.test.${EVAL_L1}${EVAL_L2} \
--encoder_tokenizer_path $ENCODER_TOKENIZER_PATH \
--moses_phrase_table ..//phrase_table/train.clean.deen.moses.phtab.json \
--freq_of_stop_words  $FREQ_OF_STOP_WORDS \
--save_dir ../data-bin/$MODEL_TYPE.eval.retriever.test.${EVAL_L1}${EVAL_L2}

EVAL_L1="ro"
EVAL_L2="en"
python dataloader.py build_dataset_v2 \
--ds_name "retrieval_with_tokenization_dataset" \
--src_lang $EVAL_L1 \
--tgt_lang $EVAL_L2 \
--src_file ../align_data/RoEn/test.clean.$EVAL_L1 \
--tgt_file ../align_data/RoEn/test.clean.$EVAL_L2 \
--align_file ../align_data/RoEn/aligned.test.${EVAL_L1}${EVAL_L2} \
--encoder_tokenizer_path $ENCODER_TOKENIZER_PATH \
--moses_phrase_table ..//phrase_table/train.clean.deen.moses.phtab.json \
--freq_of_stop_words  $FREQ_OF_STOP_WORDS \
--save_dir ../data-bin/$MODEL_TYPE.eval.retriever.test.${EVAL_L1}${EVAL_L2}

#######################################################
# Inference
#######################################################

MODEL_TYPE=ours_labse # e.g., ours_xlmr
HUMAN_TYPE=bert
SAVE_PREF=../datastore/eval_retriever/$MODEL_TYPE
MODEL_PRETRAINED_DIR=../ckpts/ccpr_labse/step_20000

$PYTHON_PATH/python eval_retriever.py encode_data \
--eval_config_path ./config/eval_retriever.yaml \
--model_type $MODEL_TYPE \
--model_pretrained_dir $MODEL_PRETRAINED_DIR \
--save_dir $SAVE_PREF \
--train_databin_prefix ../data-bin/labse.eval.retriever.train \
--eval_databin_prefix ../data-bin/labse.eval.retriever.test 


$PYTHON_PATH/python ../mytools/mytools/faiss_tools.py build_index \
--dataprefix $SAVE_PREF/data.tgt \
--index_dir $SAVE_PREF \
--index_type "flatip" \


for i in "de" "ro" "cs"
do
    $PYTHON_PATH/python ../mytools/mytools/faiss_tools.py search_single_file \
    --queryprefix $SAVE_PREF/data.src.$i \
    --index_path $SAVE_PREF/flatip.index \
    --savepath $SAVE_PREF/data.src.${i}.retrieval.out.pt \
    --index_on_gpu \
    --search_batch_size 256 \
    --search_topk 64 \
    --gpu_num 4

    $PYTHON_PATH/python eval_retriever.py retrieval_accuracy \
    --retrieval_out_path $SAVE_PREF/data.src.$i.retrieval.out.pt \
    --golden_mapping_path $SAVE_PREF/data.src.$i.golden.map \
    --topk 1
done