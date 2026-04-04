MODEL_PATH=$1

CUDA_VISIBLE_DEVICES=0,1 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longdialogue_qa_eng --start_idx 0 --stop_idx 50 &
sleep 10
CUDA_VISIBLE_DEVICES=2,3 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longdialogue_qa_eng --start_idx 50 --stop_idx 100 &
sleep 10
CUDA_VISIBLE_DEVICES=4,5 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longdialogue_qa_eng --start_idx 100 --stop_idx 150 &
sleep 10
CUDA_VISIBLE_DEVICES=6,7 python eval_commvq.py --model_name long-kv --model_path $MODEL_PATH --task longdialogue_qa_eng --start_idx 150 --stop_idx 200 &
sleep 10
