import torch
import sys
import os

# Add relevant paths
sys.path.append(os.getcwd())

from attention_aware_predictive_kv import AttentionAwarePredictiveKVCluster

def test_apkvc():
    print("Initializing APKVC Cluster...")
    # Mock some basic args to pass through
    cluster = AttentionAwarePredictiveKVCluster(
        predictor_type='linear',
        rd_threshold=0.1,
        max_anchor_interval=5,
        K_num_codebooks=4,
        V_num_codebooks=2
    )
    
    bsz, num_heads, head_dim = 1, 32, 128
    device = "cpu"
    
    # 1. Prefill
    print("Testing Prefill (S=10)...")
    q_prefill = torch.randn(bsz, num_heads, 10, head_dim)
    k_prefill = torch.randn(bsz, num_heads, 10, head_dim)
    v_prefill = torch.randn(bsz, num_heads, 10, head_dim)
    
    # Simple attendance mask
    mask_prefill = torch.zeros(bsz, 1, 10, 10)
    
    k_out, v_out = cluster.update_kv(k_prefill, q_prefill, v_prefill, mask_prefill, 1)
    print(f"Prefill Output Shapes: K={k_out.shape}, V={v_out.shape}")
    assert k_out.shape == k_prefill.shape
    assert len(cluster.entries) == 10
    
    # 2. Decoding Step 1
    print("Testing Decoding Step 1 (S=1)...")
    q1 = torch.randn(bsz, num_heads, 1, head_dim)
    k1 = torch.randn(bsz, num_heads, 1, head_dim)
    v1 = torch.randn(bsz, num_heads, 1, head_dim)
    
    k_out1, v_out1 = cluster.update_kv(k1, q1, v1, None, 1)
    print(f"Decoding Step 1 Output Shapes: K={k_out1.shape}, V={v_out1.shape}")
    assert k_out1.shape == k1.shape
    assert len(cluster.entries) == 11
    
    # 3. Decoding Step 2 (Check prediction/quantization loop)
    print("Testing Decoding Step 2...")
    q2 = torch.randn(bsz, num_heads, 1, head_dim)
    k2 = torch.randn(bsz, num_heads, 1, head_dim)
    v2 = torch.randn(bsz, num_heads, 1, head_dim)
    
    k_out2, v_out2 = cluster.update_kv(k2, q2, v2, None, 1)
    print(f"Decoding Step 2 Output Shapes: K={k_out2.shape}, V={v_out2.shape}")
    assert k_out2.shape == k2.shape
    assert len(cluster.entries) == 12
    
    # Check if we can trigger a reset by increasing seq length past max_anchor_interval
    print("Testing Anchor Reset interval...")
    current_len = len(cluster.entries)
    for i in range(10):
        q_i = torch.randn(bsz, num_heads, 1, head_dim)
        k_i = torch.randn(bsz, num_heads, 1, head_dim)
        v_i = torch.randn(bsz, num_heads, 1, head_dim)
        cluster.update_kv(k_i, q_i, v_i, None, 1)
    
    print(f"Final Count of Entries: {len(cluster.entries)}")
    print("APKVC Test Passed!")

if __name__ == "__main__":
    try:
        test_apkvc()
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
