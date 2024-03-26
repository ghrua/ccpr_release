########################################################
# Please config the following project variables
########################################################
PYTHON_PATH=/path/to/python/

# prompt config
L1=en
L2=tr
FIELD=test
phrase_max_num=32
phrase_max_dist_percent=0.75
phrase_max_context_len=100
METHOD=phrase_with_context
MODEL_TYPE=labse


# search file config
TEST_DATA_DIR=$PROJECT_ROOT_DIR/wmt16_${LANG}${Q_LANG}/
TEST_FILE_PREF=test
TEST_SRC_FILE=$TEST_DATA_DIR/$TEST_FILE_PREF.$Q_LANG
TEST_REF_FILE=$TEST_DATA_DIR/$TEST_FILE_PREF.$LANG

########################################################
# Hyper-parameters. You don't need to change them
########################################################

# searching config
# keep the same as the indexing and searching time
S_THRESHOLD=0.9
TOPK=5
PHRASE_DATA_PATH=../datastore/wmt16_newscrawl_${MODEL_TYPE}/news.20${YY}.clean.$LANG/$Q_PREFIX.$S_THRESHOLD.$Q_LANG.analysis.top${TOPK}.json
SAVE_DIR=../ft_data/

mkdir -p $SAVE_DIR

########################################################
# prepare data for inference
########################################################

$PYTHON_PATH/python translation.py prepare_alpaca_data \
--src_file_path $TEST_SRC_FILE \
--ref_file_path $TEST_REF_FILE \
--save_path $SAVE_DIR/${METHOD}.${MODEL_TYPE}.alpace.${L2}${L1}.${FIELD}.json \
--phrase_data_path $PHRASE_DATA_PATH \
--phrase_topk 1 \
--phrase_max_num $phrase_max_num \
--phrase_max_context_len $phrase_max_context_len \
--phrase_max_dist_percent $phrase_max_dist_percent \
--src_lang $L2 \
--tgt_lang $L1 \
--method $METHOD \
--field $FIELD


