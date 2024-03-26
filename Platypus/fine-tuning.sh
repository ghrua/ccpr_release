########################################################
# Please config the following project variables
########################################################

PROJECT_ROOT_DIR="../"
PYTHON_PATH=/path/to/python/
export WANDB_MODE=disabled

# model config
L1=tr
L2=en
METHOD=phrase_with_context.labse
OUTPUT_DIR=$PROJECT_ROOT_DIR/ckpts/llama2-7b.platypus.ft.mt.${METHOD}.${L1}${L2}.save100


# llama config
HF_CACHE_DIR=../huggingface/cache
BASE_MODEL=meta-llama/Llama-2-7b-hf

########################################################
# Fine-tune
########################################################


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYTHON_PATH/python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 finetune.py \
--base_model $BASE_MODEL \
--data-path $PROJECT_ROOT_DIR/ft_data/${METHOD}.alpace.${L1}${L2}.train.sample.300x6.json \
--output_dir $OUTPUT_DIR \
--batch_size 16 \
--micro_batch_size 2 \
--num_epochs 1 \
--learning_rate 0.0004 \
--cutoff_len 4096 \
--val_set_size 0 \
--lora_r 16 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--lora_target_modules '[gate_proj, down_proj, up_proj]' \
--train_on_inputs False \
--add_eos_token False \
--group_by_length False \
--prompt_template_name alpaca \
--lr_scheduler 'cosine' \
--warmup_steps 10 \
--enable_bf16 \
--save_steps 100 \
--hf_cache_dir $HF_CACHE_DIR #> log.llama2-7b.platypus.ft.mt.${METHOD}.${L1}${L2}.20240130 2>&1