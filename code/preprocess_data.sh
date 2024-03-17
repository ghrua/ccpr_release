#############################################
# Scripts on Jan 19, 2024
#############################################
PROJECT_ROOT_DIR="../"
MODEL_NAME="LaBSE"

#############################################
# Data preprocessing
# Pelease run this before all the training scripts
############################################## 

# CPU-only
L2=en
FREQ_OF_STOP_WORDS=30000

for L1 in "de" "cs" "fi" "ro" "ru" "tr"
do
    PREFIX=train
    python project_tools.py extract_moses_phrase_table \
    --src_file_path $PROJECT_ROOT_DIR/wmt16_${L1}${L2}/${PREFIX}.clean.${L1} \
    --tgt_file_path $PROJECT_ROOT_DIR/wmt16_${L1}${L2}/${PREFIX}.clean.${L2} \
    --s2t_symal_path ${PROJECT_ROOT_DIR}/moses/aligned.${PREFIX}.${L1}${L2} \
    --save_path  ${PROJECT_ROOT_DIR}/phrase_table/${PREFIX}.clean.${L1}${L2}.moses.phtab.json \
    --min_freq 1 \
    --src_lang $L1 \
    --tgt_lang $L2

    python dataloader.py build_dataset \
    --ds_name "retrieval_with_tokenization_dataset" \
    --src_lang $L1 \
    --tgt_lang $L2 \
    --project_root_dir $PROJECT_ROOT_DIR \
    --freq_of_stop_words $FREQ_OF_STOP_WORDS \
    --model_name $MODEL_NAME

done