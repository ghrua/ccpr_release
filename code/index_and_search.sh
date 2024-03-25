########################################################
# Please config the following project variables
########################################################
PROJECT_ROOT_DIR="../"
PYTHON_PATH=/path/to/python/

# model config
GPU_IDS=0,1,2,3
MODEL_TYPE=labse # e.g., labse and xlmr
MODEL_STEP=20000 # e.g., 20000, 40000
INF_MODEL_DIR=$PROJECT_ROOT_DIR/ckpts/ccpr_labse/step_$MODEL_STEP


# language config
Q_LANG=en # searching language
LANG=tr # indexing langauge e.g., de, cs, fi, ru, ro, tr

# index file config
YY=16 # year of indexing newscrawl data
NEWSCRAWL_DIR=$PROJECT_ROOT_DIR/newscrawl
NEWSCRAWL_FILE_NAME=news.20${YY}.clean.$LANG
DATASTORE_FILE=$NEWSCRAWL_DIR/$NEWSCRAWL_FILE_NAME

# search file config
TEST_DATA_DIR=$PROJECT_ROOT_DIR/wmt16_${LANG}${Q_LANG}/
TEST_FILE_PREF=test
TEST_SRC_FILE=$TEST_DATA_DIR/$TEST_FILE_PREF.$Q_LANG
TEST_REF_FILE=$TEST_DATA_DIR/$TEST_FILE_PREF.$LANG


########################################################
# Hyper-parameters. You don't need to change them
########################################################

if [[ "$MODEL_TYPE" == "xlmr" ]]; then
RET_TKNZER=$PROJECT_ROOT_DIR/huggingface/xlm-roberta-base/
else
RET_TKNZER=$PROJECT_ROOT_DIR/huggingface/LaBSE/
fi

N_SPLIT=10000000
THRESHOLD=0.7
S_THRESHOLD=0.9
DS_DIR="wmt16_newscrawl_${MODEL_TYPE}"
CONFIG=config/retriver_${MODEL_TYPE}_128_multilingual.yaml
BATCH_SIZE=256
TOPK=5
IFS=',' read -ra ADDR <<< "$GPU_IDS"
NUM_GPUS=${#ADDR[@]}

########################################################
# Data Prorcessing
########################################################

perl $PROJECT_ROOT_DIR/moses-scripts/scripts/tokenizer/remove-non-printing-char.perl < $NEWSCRAWL_DIR/news.20${YY}.${LANG}.shuffled.deduped > $DATASTORE_FILE

########################################################
# Indexing & Searching
########################################################

PREFIX=$NEWSCRAWL_FILE_NAME
Q_PREFIX=test
mode=model
PART=$NEWSCRAWL_FILE_NAME

DATASTORE_FILE=$PROJECT_ROOT_DIR/newscrawl/$NEWSCRAWL_FILE_NAME
Q_DATA_FILE_SRC_FILE=$TEST_SRC_FILE
Q_DATA_FILE_REF_FILE=$TEST_REF_FILE

INF_SAVE_DIR=$PROJECT_ROOT_DIR/datastore/${DS_DIR}/${PART}

echo "Saving to ${INF_SAVE_DIR}"
mkdir -p $INF_SAVE_DIR

# indexing newscrwal
# the indexing process is split to three setps, because on some disks the writing speed is very slow

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_PATH/python3 -m test build_inference_index $CONFIG \
--inference_model_dir $INF_MODEL_DIR \
--inference_datastore_file $DATASTORE_FILE \
--inference_save_dir $INF_SAVE_DIR \
--inference_data_mode $mode \
--inference_save_file_prefix $PREFIX \
--inference_encode_lang $LANG \
--inference_batch_size $BATCH_SIZE \
--inference_cache_size 10000000 \
--inference_tokenizer_threshold $THRESHOLD \
--data_no_rename \
# --inference_build_index # > ../logs/log.index.tokenizer.wmt16train.$PREFIX.$LANG &

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_PATH/python3 -m test build_inference_index $CONFIG \
--inference_model_dir $INF_MODEL_DIR \
--inference_datastore_file $DATASTORE_FILE \
--inference_save_dir $INF_SAVE_DIR \
--inference_data_mode $mode \
--inference_save_file_prefix $PREFIX \
--inference_encode_lang $LANG \
--inference_batch_size $BATCH_SIZE \
--inference_cache_size 10000000 \
--inference_tokenizer_threshold $THRESHOLD \
--data_no_encoding

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_PATH/python3 -m test build_inference_index $CONFIG \
--inference_model_dir $INF_MODEL_DIR \
--inference_datastore_file $DATASTORE_FILE \
--inference_save_dir $INF_SAVE_DIR \
--inference_data_mode $mode \
--inference_save_file_prefix $PREFIX \
--inference_encode_lang $LANG \
--inference_batch_size $BATCH_SIZE \
--inference_cache_size 10000000 \
--inference_tokenizer_threshold $THRESHOLD \
--data_no_encoding \
--data_no_rename \
--inference_build_index # > ../logs/log.index.tokenizer.wmt16train.$PREFIX.$LANG &

# searching

CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_PATH/python3 -m test build_inference_index $CONFIG \
--inference_model_dir $INF_MODEL_DIR \
--inference_datastore_file $Q_DATA_FILE_SRC_FILE \
--inference_save_dir $INF_SAVE_DIR \
--inference_data_mode $mode \
--inference_save_file_prefix $Q_PREFIX.$S_THRESHOLD \
--inference_encode_lang $Q_LANG \
--inference_batch_size $BATCH_SIZE \
--inference_tokenizer_n_sampling 1 \
--inference_cache_size 10000000 \
--inference_tokenizer_threshold $S_THRESHOLD \


CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_PATH/python3 -m test search_inference_index $CONFIG \
--inference_search_from bin \
--inference_save_dir $INF_SAVE_DIR \
--inference_data_mode $mode \
--inference_save_file_prefix $Q_PREFIX.$S_THRESHOLD \
--inference_encode_lang $Q_LANG \
--inference_on_gpu \
--inference_gpu_num $NUM_GPUS \
--inference_search_batch_size $BATCH_SIZE \

echo $INF_SAVE_DIR/$Q_PREFIX.$S_THRESHOLD.$Q_LANG
$PYTHON_PATH/python3 project_tools.py assemble_analysis_data \
--src_data_prefix $INF_SAVE_DIR/$Q_PREFIX.$S_THRESHOLD.$Q_LANG \
--datastore_data_prefix $INF_SAVE_DIR/$PREFIX.$LANG \
--src_corpus_data_path $Q_DATA_FILE_SRC_FILE \
--ref_corpus_data_path $Q_DATA_FILE_REF_FILE \
--datastore_corpus_data_path $DATASTORE_FILE \
--retriever_tokenizer_path $RET_TKNZER \
--model_type $MODEL_TYPE \
--chunk_size 7 \
--topk $TOPK

