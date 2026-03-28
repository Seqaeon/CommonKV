import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union

def plot_distortion_distributions(clusters, layer_names: Optional[List[str]] = None, title: str = "Distortion Distributions"):
    """
    Plots the distribution of distortion values for a set of APKVC clusters.
    
    Args:
        clusters: List of AttentionAwarePredictiveKVCluster instances.
        layer_names: Optional list of names for each layer.
        title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    
    if not isinstance(clusters, list):
        clusters = [clusters]
        
    for i, cluster in enumerate(clusters):
        data = [d for d in cluster.distortion_history if d > 0]
        if not data:
            continue
            
        label = layer_names[i] if layer_names else f"Layer {i}"
        plt.hist(data, bins=30, alpha=0.5, label=label, density=True)
        
    plt.title(title)
    plt.xlabel("Distortion (Key-Dot Proxy)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_distortion_timeline(clusters, layer_names: Optional[List[str]] = None, title: str = "Distortion over Time"):
    """
    Plots distortion values vs token index.
    
    Args:
        clusters: List of AttentionAwarePredictiveKVCluster instances.
        layer_names: Optional list of names for each layer.
        title: Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    
    if not isinstance(clusters, list):
        clusters = [clusters]
        
    for i, cluster in enumerate(clusters):
        history = cluster.distortion_history
        label = layer_names[i] if layer_names else f"Layer {i}"
        
        plt.plot(history, label=label, alpha=0.8)
        
        # Mark anchors (where distortion is 0)
        anchors = [j for j, val in enumerate(history) if val == 0]
        if anchors:
            plt.scatter(anchors, [0]*len(anchors), marker='x', s=20, alpha=0.5)

    plt.title(title)
    plt.xlabel("Token Index")
    plt.ylabel("Distortion")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def get_cluster_stats(cluster):
    """Returns summary statistics for a cluster's distortion history."""
    history = [d for d in cluster.distortion_history if d > 0]
    if not history:
        return {"count": 0, "mean": 0, "max": 0, "anchors": len(cluster.distortion_history)}
        
    return {
        "count": len(cluster.distortion_history),
        "compressed_count": len(history),
        "anchor_count": len(cluster.distortion_history) - len(history),
        "mean_distortion": np.mean(history),
        "max_distortion": np.max(history),
        "std_distortion": np.std(history)
    }

def get_all_clusters(model):
    """
    Traverses a model to find all AttentionAwarePredictiveKVCluster instances.
    Works for Llama and Mistral models modified by CommonKV.
    """
    clusters = []
    layer_names = []
    
    # Try to find layers in model.model.layers (standard Llama/Mistral)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        print("[Viz] Could not find layers in model.")
        return [], []

    for i, layer in enumerate(layers):
        # Look for kv_cluster in self_attn
        attn = None
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
        elif hasattr(layer, "attention"):
            attn = layer.attention
            
        if attn and hasattr(attn, "kv_cluster"):
            from attention_aware_predictive_kv import AttentionAwarePredictiveKVCluster
            if isinstance(attn.kv_cluster, AttentionAwarePredictiveKVCluster):
                clusters.append(attn.kv_cluster)
                layer_names.append(f"Layer {i}")
                
    return clusters, layer_names
