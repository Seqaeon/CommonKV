import os
import json
import random
import argparse
import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List
from svd_utils import get_rank


# Default context length list if nothing is specified via CLI
DEFAULT_CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

datasets = [
            "niah_single_1",
            "niah_single_2",
            "niah_multikey_1",
            "niah_multikey_2",
            "niah_multiquery",
            "niah_multivalue",
            "qa1",
            "qa2",
            "fwe",
            "vt"
]

dataset2maxlen = {
    "niah_single_1": 64,
    "niah_single_2": 64,
    "niah_single_3": 64,
    "niah_multikey_1": 64,
    "niah_multikey_2": 64,
    "niah_multikey_3": 64,
    "niah_multiquery": 64,
    "niah_multivalue": 64,
    "qa1": 64,
    "qa2": 64,
    "cwe": 64,
    "fwe": 64,
    "vt": 64
}


model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 31500
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt


def resolve_effective_rank(head_wise_ranks, layer_idx, requested_rank, kv_width, warn_prefix="", log_decision=False):
    rank_entry = head_wise_ranks.get(f"model.layers.{layer_idx}.self_attn.v_proj")
    metadata_missing = not rank_entry
    fallback_rank = max(1, min(requested_rank, kv_width))
    raw_rank = fallback_rank if metadata_missing else rank_entry[0]
    effective_rank = max(1, min(raw_rank, kv_width))
    if effective_rank > kv_width:
        print(f"{warn_prefix}[WARN] effective_rank={effective_rank} exceeds kv_width={kv_width}. This should never happen.")
    if log_decision:
        print(
            f"{warn_prefix}layer={layer_idx}, requested_rank={requested_rank}, kv_width={kv_width}, "
            f"effective_rank={effective_rank}, rank_source={'fallback' if metadata_missing else 'metadata'}"
        )
    return effective_rank, metadata_missing


def canonicalize_method_name(method_name: str) -> str:
    canonical_map = {
        "fullkv": "FullKV",
        "snapkv": "SnapKV",
        "streamingllm": "StreamingLLM",
        "h2o": "H2O",
        "pyramidkv": "PyramidKV",
        "l2norm": "L2Norm",
        "cam": "CAM",
        "think": "ThinK",
        "palu": "Palu",
        "minicache": "MiniCache",
    }
    return canonical_map.get(method_name.lower(), method_name)



def main(args):
    

    print("Loading data...")
    
    test_data = []
    prompt_list = []
    input_list = []
    outputs_list: List[List[str]] = [] # List of List
    length_list = []
    index_list = []
    
    input_max_len = 0
    model_path = args.model_path.lower()
    model_max_len = 8192 # Fallback config
    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
    
    # Parity with run_longbench.py: Cap prefill length for OOM-prone methods
    OOM_PRONE_METHODS = {
    "snapkv", "pyramidkv", "h2o", "cam", "l2norm", "adakv", "headkv", "streamingllm", "think", "palu", "minicache", "custom"
}
    if args.method and args.method.lower() in OOM_PRONE_METHODS and model_max_len > args.max_prefill_tokens_for_custom_methods:
        print(
            f"[WARN] Capping prefill length from {model_max_len} to {args.max_prefill_tokens_for_custom_methods} "
            f"for custom method '{args.method}' to avoid OOM."
        )
        model_max_len = args.max_prefill_tokens_for_custom_methods
    
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:

            example = json.loads(line)
            
            # Robust key mapping for different RULER formats
            # Format A: length, input, outputs
            # Format B: context, question, answer
            
            if "length" not in example:
                # If length is missing, we approximate or use 0
                example["length"] = len(example.get("context", "")) + len(example.get("question", ""))
            
            if "input" not in example:
                example["input"] = example.get("context", "") + example.get("question", "")
            
            if "outputs" not in example:
                ans = example.get("answer", "")
                example["outputs"] = [ans] if not isinstance(ans, list) else ans
                
            length = example["length"]
            if length > input_max_len: 
                input_max_len = length

            prompt = example["input"] #TODO tokenizer.apply_chat_template ?
            if "llama2" in args.model_path.lower():
                prompt = build_chat(prompt)
            example["prompt"] = prompt
                
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    for idx, example in enumerate(test_data):
        prompt_list.append(example["prompt"])
        input_list.append(example["input"])
        outputs_list.append(example["outputs"])
        length_list.append(example["length"])
        index_list.append(example.get("index", idx))

    print("Finish loading model and tokenizer")
    # Restore original casing for model_name and handle trailing slashes
    model_name = args.model_path.rstrip("/").split("/")[-1]

    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", str(args.context_length), args.dataset), exist_ok=True)
    output_dir = os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", str(args.context_length), args.dataset)
    json_output_path = os.path.join(output_dir, f"{args.method}.json")
    pretty_json_output_path = os.path.join(output_dir, f"{args.method}.pretty.json")
    jsonl_output_path = os.path.join(output_dir, f"{args.method}.jsonl")
    predictions = []
    
    for i in tqdm(range(0, len(prompt_list), args.eval_batch_size)):
        if args.steps != -1 and i >= args.steps: break
        batch_prompts = prompt_list[i:i+args.eval_batch_size]
        batch_inputs = input_list[i:i+args.eval_batch_size]
        batch_answers = outputs_list[i:i+args.eval_batch_size]
        batch_lengths = length_list[i:i+args.eval_batch_size]
        
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
            
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

        if args.max_capacity_prompts != -1:
            max_capacity_prompts = args.max_capacity_prompts
        elif args.max_capacity_prompts_ratio != -1:
            max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)
        
        # Parity with run_longbench.py: Automatic Palu rank calculation
        if args.method.lower() == 'palu':
            head_dim = getattr(model.config, 'head_dim', 128)
            palu_rank = int(args.pruning_ratio * head_dim)
            args.rank = palu_rank
        
        
        if args.method.lower() not in ["fullkv", "ours", "commonkv"] :
            if args.method.lower() in ["snapkv","pyramidkv","h2o","cam", "l2norm", "think", "palu", "minicache"]:
                window_sizes = 8
            elif args.method.lower() in ["streamingllm"]:
                window_sizes = max_capacity_prompts - 4

            kernel_sizes = 7
            pooling = "maxpool"

            ratio = args.pruning_ratio
            recent_size = args.recent_size

            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            if not isinstance(ratio, list):
                ratio = [ratio] * layers
            if not isinstance(recent_size, list):
                recent_size = [recent_size] * layers

            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling
                model.model.layers[i].self_attn.config.merge = args.merge
                model.model.layers[i].self_attn.config.floor = args.floor
                model.model.layers[i].self_attn.config.ratio = ratio[i]
                model.model.layers[i].self_attn.config.recent_size = recent_size[i]
                model.model.layers[i].self_attn.config.rank = args.rank

        context_length = batch_input_ids.shape[-1]
        if args.quant_method == None:        
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id]
            )
        else:
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id],
                cache_implementation="quantized", 
                cache_config={"nbits": args.nbits, "backend": "HQQ","device":"cuda","residual_length":output_max_len,"axis_key":1,"q_group_size":64},
            )

        batch_outputs = tokenizer.batch_decode(output[:, context_length:], skip_special_tokens=True)
        batch_generations = batch_outputs

        torch.cuda.empty_cache()
        
        for j in range(len(batch_prompts)):
            
            example = {}
            example["prompt"] = batch_prompts[j]
            example["input"] = batch_inputs[j]
            example["answers"] = batch_answers[j]
            example["pred"] = batch_generations[j]
            example["length"] = batch_lengths[j]

            predictions.append(example)

    with open(json_output_path, "w") as fout_json:
        for example in predictions:
            fout_json.write(json.dumps(example, ensure_ascii=False) + "\n")
    with open(pretty_json_output_path, "w") as fout_pretty:
        json.dump(predictions, fout_pretty, ensure_ascii=False, indent=2)
    with open(jsonl_output_path, "w") as fout_jsonl:
        for example in predictions:
            fout_jsonl.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--quant_method",type=str,default=None,choices=["kivi","kvquant"])
    parser.add_argument("--nbits", type=int, default=8, help="")
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--max_prefill_tokens_for_custom_methods", type=int, default=2048, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--pruning_ratio", type=float, default=0.4, help="")
    parser.add_argument("--recent_size", type=int, default=32, help="")
    parser.add_argument("--merge", action="store_true", help="Whether to merge KV states in certain methods.")
    parser.add_argument("--floor", type=float, default=0.2, help="Floor for importance scoring.")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--max_datasets", type=int, default=-1, help="maximum number of datasets to evaluate.")
    parser.add_argument("--rank", type=int, default=1024, help="rank of up and down matrix")
    parser.add_argument("--layer_step", type=int, default=2, help="how many layers connect to one")
    parser.add_argument("--context_lengths", type=int, nargs="+", default=None, help="Context lengths to evaluate. Defaults to DEFAULT_CONTEXT_LENGTHS.")
    parser.add_argument(
        "--require_head_wise_ranks",
        action="store_true",
        help="Fail fast if head-wise rank metadata is missing when using CommonKV.",
    )
    
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    
    args = parser.parse_args()
    if args.method and args.method.lower() == "ours":
        print("[INFO] Method alias 'ours' detected; normalizing to 'commonkv'.")
        args.method = "commonkv"
    if args.method:
        args.method = canonicalize_method_name(args.method)
    
    set_seed(args.seed)
    if args.quant_method == "kvquant":
        from commonkv.quantcache import KVQuantizedCache
        from transformers import cache_utils
        cache_utils.HQQQuantizedCache = KVQuantizedCache
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )


    from commonkv.monkeypatch import replace_llama,replace_mistral
    replace_llama(args.method.lower())
    replace_mistral(args.method.lower())
    
    config = AutoConfig.from_pretrained(args.model_path, use_cache=args.use_cache)
    config.rank = args.rank
    config.layer_step = args.layer_step
    if args.method.lower() in ['ours', 'commonkv']:
        config.head_wise_ranks = get_rank(args.model_path)
    else:
        config.head_wise_ranks = {}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )

    if args.method.lower() in ['ours', 'commonkv']:
        RANK = args.rank
        layer_step = args.layer_step

        num_layers = len(model.model.layers)
        head_wise_ranks = get_rank(args.model_path)
        if not head_wise_ranks:
            message = (
                "[WARN] CommonKV rank metadata is missing/empty. Using fallback rank policy: "
                "effective_rank=max(1, min(requested_rank, kv_width))."
            )
            if args.require_head_wise_ranks:
                raise ValueError(message.replace("[WARN]", "[ERROR]"))
            print(message)



        for start_idx in tqdm(range(0, num_layers, layer_step)):
            end_idx = min(start_idx + layer_step, num_layers)
            layers = [model.model.layers[i].self_attn for i in range(start_idx, end_idx)]

            k_weights = [layer.k_proj.weight.data.float() for layer in layers]
            k_cat = torch.cat(k_weights, dim=0)
            v_weights = [layer.v_proj.weight.data.float() for layer in layers]
            v_cat = torch.cat(v_weights, dim=0)
            kv_cat = torch.cat([k_cat, v_cat], dim=0)

            KVU, KVS, KVVt = torch.linalg.svd(kv_cat, full_matrices=False)

            offset = 0
            for i, layer in enumerate(layers):
                current_layer_num = start_idx + i
                kv_width = layer.k_proj.weight.shape[0]
                current_rank, _ = resolve_effective_rank(
                    head_wise_ranks=head_wise_ranks,
                    layer_idx=current_layer_num,
                    requested_rank=args.rank,
                    kv_width=kv_width,
                    warn_prefix="[CommonKV][RULER] ",
                    log_decision=True,
                )

                cur_KVS = KVS[:current_rank]
                sqrtSigma = torch.sqrt(torch.diag(cur_KVS))
                KV_reduced = (KVU[:, :current_rank] @ sqrtSigma)
                KVVt_reduced = sqrtSigma @ KVVt[:current_rank]

                D_out = k_weights[i].shape[0]
                layer.k_down_proj.weight.data = KV_reduced[offset:offset + D_out].half()
                layer.k_up_proj.weight.data = KVVt_reduced.to(torch.float16)
                offset += D_out
                layer.k_proj = None
            for i, layer in enumerate(layers):
                current_layer_num = start_idx + i
                kv_width = layer.v_proj.weight.shape[0]
                current_rank, _ = resolve_effective_rank(
                    head_wise_ranks=head_wise_ranks,
                    layer_idx=current_layer_num,
                    requested_rank=args.rank,
                    kv_width=kv_width,
                    warn_prefix="[CommonKV][RULER] ",
                    log_decision=True,
                )

                cur_KVS = KVS[:current_rank]
                sqrtSigma = torch.sqrt(torch.diag(cur_KVS))
                KV_reduced = (KVU[:, :current_rank] @ sqrtSigma)
                KVVt_reduced = sqrtSigma @ KVVt[:current_rank]

                D_out = v_weights[i].shape[0]
                layer.v_down_proj.weight.data = KV_reduced[offset:offset + D_out].half()
                layer.v_up_proj.weight.data = KVVt_reduced.to(torch.float16)
                offset += D_out
                layer.v_proj = None

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    save_dir = args.save_dir
    max_capacity_prompts = args.max_capacity_prompts
    
    # Determine which context lengths to run
    if args.context_lengths:
        target_context_lengths = args.context_lengths
    else:
        target_context_lengths = DEFAULT_CONTEXT_LENGTHS
    
    if args.dataset in [None, "all", ""]:
        if args.max_datasets != -1:
            datasets = datasets[:args.max_datasets]
            
        for context_length in target_context_lengths:
            for idx, dataset in enumerate(datasets):
                print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx+1}/{len(datasets)} - Context {context_length}")
                args.context_length = context_length
                args.dataset = dataset
                args.data_file = f"data/RULER/{context_length}/{args.dataset}.jsonl"
                main(args)
    else:
        # Just run the single dataset requested by the command line
        for context_length in target_context_lengths:
            args.context_length = context_length
            print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {args.dataset} - Context {context_length}")
            if not args.data_file:
                args.data_file = f"data/RULER/{context_length}/{args.dataset}.jsonl"
            main(args)
