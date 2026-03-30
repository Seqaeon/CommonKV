import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

COLORS = {
    "FullKV":    "#888780",
    "KIVI-int4": "#3B8BD4",
    "KIVI-int2": "#185FA5",
    "APKVC-identity": "#9E6AC4",
    "APKVC-linear":   "#6E3A9A",
}

def plot1_compression_vs_length(results_by_method: dict, task_name: str,
                                 save_path: str = None):
    """
    Plot 1: Compression ratio vs tokens generated.

    results_by_method: {method_name: {"snapshots": [{"tokens_generated": int,
                                                      "compression_ratio_mean": float,
                                                      "compression_ratio_std": float}]}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, results in results_by_method.items():
        snaps = results["snapshots"]
        xs = [s["tokens_generated"] for s in snaps]
        ys = [s["compression_ratio_mean"] for s in snaps]
        stds = [s.get("compression_ratio_std", 0) for s in snaps]
        color = COLORS.get(method_name, "#333")

        ax.plot(xs, ys, label=method_name, color=color, linewidth=2, marker="o", ms=5)
        ax.fill_between(xs,
                         [y - s for y, s in zip(ys, stds)],
                         [y + s for y, s in zip(ys, stds)],
                         alpha=0.12, color=color)

    ax.axhline(y=1.0, color=COLORS["FullKV"], linestyle="--", linewidth=1.2,
               label="FullKV (1.0)")
    ax.set_xlabel("Tokens generated", fontsize=12)
    ax.set_ylabel("Compression ratio  (lower = more compressed)", fontsize=12)
    ax.set_title(f"Compression ratio vs generation length — {task_name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # Annotate: APKVC improves over time, KIVI stays flat
    if "APKVC-identity" in results_by_method and "KIVI-int4" in results_by_method:
        ax.annotate("APKVC improves →", xy=(0.62, 0.35), xycoords="axes fraction",
                    fontsize=9, color=COLORS["APKVC-identity"])
        ax.annotate("KIVI stays flat →", xy=(0.62, 0.55), xycoords="axes fraction",
                    fontsize=9, color=COLORS["KIVI-int4"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot2_pareto_frontier(pareto_data: list, save_path: str = None):
    """
    Plot 2: Quality (perplexity or ROUGE-L) vs compression ratio.

    pareto_data: list of dicts:
      {"method": str, "compression_ratio": float, "perplexity": float,
       "rouge_l": float, "config_label": str}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel, invert in [
        (axes[0], "perplexity",      "Perplexity (lower = better)", True),
        (axes[1], "rouge_l",         "ROUGE-L (higher = better)",   False),
    ]:
        plotted_methods = set()
        for point in pareto_data:
            m = point["method"]
            color = COLORS.get(m, "#333")
            label = m if m not in plotted_methods else None
            ax.scatter(point["compression_ratio"], point[metric],
                       color=color, s=80, label=label, zorder=3)
            ax.annotate(point.get("config_label", ""),
                        (point["compression_ratio"], point[metric]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=color)
            plotted_methods.add(m)

        # Draw Pareto frontier for APKVC points
        apkvc_points = [(p["compression_ratio"], p[metric])
                        for p in pareto_data if "APKVC" in p["method"]]
        if apkvc_points:
            apkvc_points.sort(key=lambda x: x[0])
            # Pareto: no point dominated on both axes
            frontier = []
            best = float("inf") if invert else float("-inf")
            for cr, q in apkvc_points:
                if (invert and q < best) or (not invert and q > best):
                    frontier.append((cr, q))
                    best = q
            if frontier:
                fx, fy = zip(*frontier)
                ax.step(fx, fy, where="post", color=COLORS["APKVC-linear"],
                        linewidth=1.5, linestyle="--", alpha=0.6, label="APKVC Pareto")

        ax.set_xlabel("Compression ratio  (lower = more compressed)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_title(ylabel, fontsize=12)

    fig.suptitle("Quality vs compression ratio (Pareto frontier)", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot3_vram_over_turns(turn_results_by_method: dict, gpu_limit_gb: float = 22.0,
                           save_path: str = None):
    """
    Plot 3: VRAM (GB) over conversation turns.

    turn_results_by_method: {method_name: [{"turn": int, "vram_gb": float}]}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, turns in turn_results_by_method.items():
        xs = [t["turn"] for t in turns]
        ys = [t["vram_gb"] for t in turns]
        color = COLORS.get(method_name, "#333")
        ax.plot(xs, ys, label=method_name, color=color, linewidth=2, marker="o", ms=5)

    ax.axhline(y=gpu_limit_gb, color="#E24B4A", linestyle="--", linewidth=1.5,
               label=f"GPU limit ({gpu_limit_gb} GB)")

    ax.set_xlabel("Conversation turn", fontsize=12)
    ax.set_ylabel("Peak VRAM (GB)", fontsize=12)
    ax.set_title("VRAM usage over multi-turn conversation", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
