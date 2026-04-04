MODEL_PATH=$1

CUDA_VISIBLE_DEVICES=0,1 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task code_debug --start_idx 0 --stop_idx 98 &
sleep 10
CUDA_VISIBLE_DEVICES=2,3 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task code_debug --start_idx 98 --stop_idx 197 &
sleep 10
CUDA_VISIBLE_DEVICES=4,5 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task code_debug --start_idx 197 --stop_idx 295 &
sleep 10
CUDA_VISIBLE_DEVICES=6,7 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task code_debug --start_idx 295 --stop_idx 394 &
sleep 10
