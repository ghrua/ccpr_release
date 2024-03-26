########################################################
# Please config the following project variables
########################################################
PYTHON_PATH=/path/to/python/

METHOD=phrase_with_context
MODEL_TYPE=labse
L1=en
L2=tr
FIELD=test
GPU_ID=0
HF_CACHE_DIR=../huggingface/hub
CKPT_DIR=../ckpts/llama2-7b.platypus.ft.mt.${METHOD}.${MODEL_TYPE}.enxx/checkpoint-100


########################################################
# Hyper-parameters. You don't need to change them
########################################################

FNAME=${METHOD}.${MODEL_TYPE}.alpace.${L1}${L2}.${FIELD}
INPUT_DATA=../ft_data/$FNAME.json
SAVE_DATA=../ft_out/$FNAME.hypo.json

mkdir -p ../ft_out/

########################################################
# Inference & report score
########################################################

# LOG_PATH=log.$FNAME.llama2-7b
# echo $LOG_PATH
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_PATH/python inference.py \
--base_model meta-llama/Llama-2-7b-hf \
--lora_weights $CKPT_DIR \
--input_data_path $INPUT_DATA \
--save_data_path $SAVE_DATA \
--max_new_tokens 256 \
--temperature 1.0 \
--top_p 0.1 \
--do_sample False \
--field $FIELD \
--hf_cache_dir $HF_CACHE_DIR #> $LOG_PATH 2>&1


# echo $LOG_PATH.score
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_PATH/python translation.py report_score \
--trans_dump_file $SAVE_DATA \
--src_file_path ../wmt16_${L2}${L1}/${FIELD}.clean.${L1} \
--ref_file_path ../wmt16_${L2}${L1}/${FIELD}.clean.${L2} \
--tgt_lang $L2 \
--data_format alpaca \
--field $FIELD #> $LOG_PATH.score 2>&1