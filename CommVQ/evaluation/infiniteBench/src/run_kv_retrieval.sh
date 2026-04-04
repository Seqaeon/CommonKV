MODEL_PATH=$1

CUDA_VISIBLE_DEVICES=0,1 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task kv_retrieval --start_idx 0 --stop_idx 125 &
sleep 10
CUDA_VISIBLE_DEVICES=2,3 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task kv_retrieval --start_idx 125 --stop_idx 250 &
sleep 10
CUDA_VISIBLE_DEVICES=4,5 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task kv_retrieval --start_idx 250 --stop_idx 375 &
sleep 10
CUDA_VISIBLE_DEVICES=6,7 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task kv_retrieval --start_idx 375 --stop_idx 500 &
sleep 10
