MODEL_PATH=$1

CUDA_VISIBLE_DEVICES=0,1 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longbook_qa_eng --start_idx 0 --stop_idx 87 &
sleep 10
CUDA_VISIBLE_DEVICES=2,3 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longbook_qa_eng --start_idx 87 --stop_idx 175 &
sleep 10
CUDA_VISIBLE_DEVICES=4,5 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longbook_qa_eng --start_idx 175 --stop_idx 263 &
sleep 10
CUDA_VISIBLE_DEVICES=6,7 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longbook_qa_eng --start_idx 263 --stop_idx 351 &
sleep 10
