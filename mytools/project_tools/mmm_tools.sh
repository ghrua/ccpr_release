DEST=/apdcephfs/share_916081/redmondliu/intern_data/alanili/
# GROUPED_IDX=$DEST/data/CC3M/CC3M_repr/cls_12.group.idx
# CAPTION_DATA=$DEST/data/CC3M/train.caption.txt
GROUPED_IDX=$DEST/data/CC3M/CC3M_repr/cls_12_vision.clip.group.idx
CAPTION_DATA=$DEST/data/CC3M/success.pt
MATCH_SCORE_DATA=$DEST/data/CC3M/match_score.pt
FEWSHOT_POOL=/apdcephfs/share_916081/redmondliu/intern_data/alanili/mytools/project_tools/data_out/fewshot_pool.v3.3.json


# python -m mmm_tools generate_mmmchat_mp $GROUPED_IDX $CAPTION_DATA "./data_out/cc3m.cls_12_vision.clip.gpt4.v6" --model gpt-4 --nproc 1 --gen_num 100000 --match_score_path $MATCH_SCORE_DATA --fewshot_pool_file $FEWSHOT_POOL

# python -m mmm_tools generate_from_old \
# --old_data_path "./data_out/cc3m.cls_12_vision.clip.gpt3_5.v2.0.data.json" \
# --save_path "./data_out/cc3m.cls_12_vision.clip.gpt3.5.v4.data.json" \
# --model gpt-3.5-turbo \
# --fewshot_pool_file "./data_out/fewshot_pool.json"

# python -m mmm_tools generate_from_old \
# --old_data_path "./data_out/cc3m.cls_12_vision.clip.gpt3_5.v2.0.data.json" \
# --save_path "./data_out/cc3m.cls_12_vision.clip.gpt4.v4.data.json" \
# --model gpt-4 \
# --fewshot_pool_file "./data_out/fewshot_pool.json"


# python -m mmm_tools generate_from_old \
# --old_data_path "./data_out/cc3m.cls_12_vision.clip.gpt3_5.v2.0.data.json" \
# --save_path "./data_out/cc3m.cls_12_vision.clip.gpt3.5-16k.v4.data.json" \
# --model "gpt-3.5-turbo-16k" \
# --fewshot_pool_file "./data_out/fewshot_pool.json"

# python -m mmm_tools clean_fewshot_pool \
# --fewshot_pool_path ./data_out/fewshot_pool.v3.1.json \
# --annotation_file_path ./data_out/fewshot_pool.v3.1.annotation.csv \
# --save_path ./data_out/fewshot_pool.v3.3.json

# python3 openai_utils.py mim \
# --prompt_file_path ./data_out/cc3m.cls_12_vision.clip.gpt4.v6.0.data.json \
# --save_dir ./mim_out \
# --start_id 0 \
# --end_id 1500 \

# python3 openai_utils.py mim \
# --prompt_file_path ./data_out/train.s2.v1.dupimg.0.100.json \
# --save_dir ./debug_dup_out \
# --start_id 0 \
# --end_id 10 \
# --batch_size 10 \

python -m mmm_tools clean_mim_data \
--data_dir ./lywang_out/ \
--start_id 6846 \
--end_id 14564 \
--is_single_file