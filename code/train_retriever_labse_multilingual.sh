#############################################
# scripts for Jan 19, 2024
#############################################
PROJECT_ROOT_DIR="../"
MODEL_NAME="LaBSE"


#############################################
# Model Training
############################################## 
# Shared Hyper-parameters

TKZ_LOSS_WEIGHT=1.0
DROPOUT_RATE=0.2
MASK_PERCENT=0.0
PAIR_PERCENT=0.7

CHECKPOINTING_STEPS=5000
MAX_TRAIN_STEPS=30000
BATCH_SIZE=64

#############################################
# Train with 8 GPU
#############################################

L1=cs,de,fi,ro,ru,tr
L2=en,en,en,en,en,en

OUTPUT_DIR=$PROJECT_ROOT_DIR/ckpts/retriever_labse_multilingual/

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_retriever.py --project_config config/retriver_labse_128_multilingual.yaml --override_args "{'output_dir': '${OUTPUT_DIR}', 'max_train_steps': ${MAX_TRAIN_STEPS}, 'tkz_loss_weight': ${TKZ_LOSS_WEIGHT}, 'dropout_rate': ${DROPOUT_RATE}, 'paired_data_percent': ${PAIR_PERCENT}, 'use_single_pos': False, 'masked_paired_data_percent': ${MASK_PERCENT}, 'checkpointing_steps': ${CHECKPOINTING_STEPS}, 'batch_size': ${BATCH_SIZE}, 'data_pretrained_dir_prefix': '${PROJECT_ROOT_DIR}/data-bin/labse.moses.withtkz.retrival.max128.fromgiza', 'src_lang': '${L1}', 'tgt_lang': '${L2}'}"
