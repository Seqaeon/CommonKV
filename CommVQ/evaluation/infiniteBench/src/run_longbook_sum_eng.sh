MODEL_PATH=$1

CUDA_VISIBLE_DEVICES=0,1 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longbook_sum_eng --start_idx 0 --stop_idx 25 &
sleep 10
CUDA_VISIBLE_DEVICES=2,3 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longbook_sum_eng --start_idx 25 --stop_idx 51 &
sleep 10
CUDA_VISIBLE_DEVICES=4,5 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longbook_sum_eng --start_idx 51 --stop_idx 77 &
sleep 10
CUDA_VISIBLE_DEVICES=6,7 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longbook_sum_eng --start_idx 77 --stop_idx 103 &
sleep 10
