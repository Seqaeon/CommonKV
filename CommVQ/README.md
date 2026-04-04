# CommVQ: Commutative Vector Quantization for KV Cache Compression

[[Paper](https://arxiv.org/abs/2506.18879)] [[Hugging Face Models](https://huggingface.co/collections/senfu/commvq-68412ebc14e0f9cdfffb7172)]

This repository contains the official implementation of **CommVQ**, a method for **memory-efficient** and **long-context** inference through KV cache quantization with learned codebooks. It achieves strong performance across a wide range of benchmarks while significantly reducing memory overhead.


## Table of Contents

- [News](#news)
- [Model Checkpoints](#model-checkpoints)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
  - [Longbench](#longbench)
  - [Infinitebench](#infinitebench)
  - [NIAH](#niah)
- [Memory Measurement](#memory-measurement)
- [Citation](#citation)

## News

- **[June, 2025]**: Released code and model weights.
- **[May, 2025]**: CommVQ is accepted to **ICML 2025**! See you in Vancouver, BC.

## Model Checkpoints

We release the following LLaMA-3.1 8B checkpoints with **CommVQ 1-bit** and **2-bit** compression. Both **value codebooks** and **key codebooks** are provided below. The **value codebooks** are used together with the original (unchanged) model weights.

| Model Variant | Value Codebook | Key Codebook |
|---------------|-------|----------|
| LLaMA-3.1 8B + CommVQ 1-bit | [🤗 Hugging Face](https://huggingface.co/senfu/Llama-3.1-8B-Instruct-CommVQ-1bit) | [🤗 Hugging Face](https://huggingface.co/senfu/Llama-3.1-8B-Instruct-CommVQ-1bit-codebook) |
| LLaMA-3.1 8B + CommVQ 2-bit | [🤗 Hugging Face](https://huggingface.co/senfu/Llama-3.1-8B-Instruct-CommVQ-2bit) | [🤗 Hugging Face](https://huggingface.co/senfu/Llama-3.1-8B-Instruct-CommVQ-2bit-codebook) |


## Installation

```python
conda create -n commvq python=3.10
conda activate commvq
pip install -e .
pip install flash-attn --no-build-isolation
```

## Training

```bash
cd training

# Step 1: Collect KV cache
bash collect_kv.sh

# Step 2: Prepare scaling factors
python make_scale.py

# Step 3: Train the codebook for key cache
bash quantize_key_cache.sh

# Step 4: Train the codebook for value cache
bash finetune/llama3.1_8b_int1.sh
```

## Evaluation

### Longbench

```python
cd evaluation/longbench
python pred.py --model $CHECKPOINT
python eval.py --model $RESULT_DIR
```

### Infinitebench

```python
cd evaluation/infiniteBench/src
# Download the evaluation datasets
bash scripts/download_dataset.sh
# Evaluate each tasks
bash run_passkey.sh
# Merge all results in each task into one jsonl file
cat ../results/commvq/preds_passkey_*.jsonl > ../results/commvq/preds_passkey.jsonl
# Compute the task score
python compute_scores.py --task all --model_name commvq
```


### NIAH

```python
cd evaluation/niah
bash run.sh $CHECKPOINT
```

## Memory Measurement

We implement Triton-based kernels to further optimize memory usage and enable real memory savings with CommVQ.
(Currently supports LLaMA-3.1 8B with 1-bit quantization; ongoing development for broader model support.)

```python
cd evaluation/memory_measurement
pip install -e ../../transformers_triton_infer
bash eval_memory.sh $CHECKPOINT
```

## Citation

If you find **CommVQ** useful in your research or applications, please consider citing:

```bibtex
@inproceedings{li2025commvq,
  title = {CommVQ: Commutative Vector Quantization for KV Cache Compression},
  author = {Junyan Li and Yang Zhang and Muhammad Yusuf Hassan and Talha Chafekar and Tianle Cai and Zhile Ren and Pengsheng Guo and Binazir Karimzadeh and Colorado J Reed and Chong Wang and Chuang Gan},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year = {2025}
}
```
