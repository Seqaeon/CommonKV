import subprocess
import os

wrapper = """
import torch
from attention_aware_predictive_kv import AttentionAwarePredictiveKVCluster, APKVCConfig

torch.manual_seed(42)

bsz, num_heads, head_dim = 2, 4, 64
q_len = 20

# Create dummy prefill
q_prefill = torch.randn(bsz, num_heads, 10, head_dim)
k_prefill = torch.randn(bsz, num_heads, 10, head_dim)
v_prefill = torch.randn(bsz, num_heads, 10, head_dim)

cluster = AttentionAwarePredictiveKVCluster(predictor_type='attention', rd_threshold=0.5, max_anchor_interval=5)

cluster.update_kv(k_prefill, q_prefill, v_prefill, None, 1)
print("Prefill complete, anchor buffer size:", cluster.anchor_buffer_K.shape)

for i in range(10):
    q_i = torch.randn(bsz, num_heads, 1, head_dim)
    k_i = k_prefill[:, :, -1:] + 0.1 * torch.randn(bsz, num_heads, 1, head_dim)
    v_i = v_prefill[:, :, -1:] + 0.1 * torch.randn(bsz, num_heads, 1, head_dim)
    cluster.update_kv(k_i, q_i, v_i, None, 1)

print("Decoding complete")
anchors = sum([1 for e in cluster.entries if e['is_anchor']])
compressed = len(cluster.entries) - anchors
print(f"Total entries: {len(cluster.entries)}, Anchors: {anchors}, Compressed: {compressed}")
"""
with open('test_attn.py', 'w') as f:
    f.write(wrapper)
try:
    subprocess.run("source ~/miniconda3/bin/activate && python3 test_attn.py", shell=True, executable='/bin/bash')
except Exception as e:
    print(e)
