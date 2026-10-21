# sobol_plots.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = pd.read_csv("sobol_indices.csv")


def plot_bars(target, top_k=15, path=None):
    """S1 vs ST per parameter for one output, with 95% bootstrap CIs."""
    sub = res[res.output == target].sort_values("ST", ascending=False).head(top_k)
    sub = sub.iloc[::-1]                                  # largest at the top
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(sub) + 1.5))
    ax.barh(y + 0.2, sub["ST"], height=0.4, xerr=sub["ST_conf"], color="#5C6BC0",
            label="Total-order (ST)", capsize=2)
    ax.barh(y - 0.2, sub["S1"], height=0.4, xerr=sub["S1_conf"], color="#FB8C00",
            label="First-order (S1)", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["param"], fontsize=8)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Sobol index")
    ax.set_title(f"Sobol sensitivity: {target}\nST - S1 gap = interaction effects", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    path = path or f"sobol_bars_{target.replace('|', '_').replace('-', '_minus_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("saved", path)


def plot_heatmap(outputs, top_k=20, path="sobol_heatmap.png"):
    """ST for the top parameters across several outputs, to compare rankings."""
    piv = res[res.output.isin(outputs)].pivot(index="param", columns="output", values="ST")
    piv = piv[outputs]
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).head(top_k).index]
    fig, ax = plt.subplots(figsize=(1.6 * len(outputs) + 3, 0.35 * len(piv) + 2))
    im = ax.imshow(piv.clip(lower=0).values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(outputs)))
    ax.set_xticklabels(outputs, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=7)
    fig.colorbar(im, ax=ax, label="Total-order index (ST)")
    ax.set_title("Total-order Sobol indices across outputs", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("saved", path)


def plot_interaction_gap(target, top_k=15, path=None):
    """ST - S1 per parameter: how much of its influence comes through interactions."""
    sub = res[res.output == target].copy()
    sub["gap"] = sub["ST"] - sub["S1"]
    sub = sub.sort_values("gap", ascending=False).head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(sub) + 1.5))
    ax.barh(sub["param"], sub["gap"], color="#EC407A")
    ax.set_xlabel("ST - S1 (interaction contribution)")
    ax.set_title(f"Interaction strength: {target}", fontsize=10)
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    path = path or f"sobol_interactions_{target.replace('|', '_').replace('-', '_minus_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("saved", path)


if __name__ == "__main__":
    targets = ["pain_auc|FUNCTIONALITY-RAW_PAIN",
               "addiction_auc|FUNCTIONALITY-RAW_PAIN",
               "time_severe_pain|FUNCTIONALITY-RAW_PAIN"]
    for t in targets:
        plot_bars(t)
        plot_interaction_gap(t)
    plot_heatmap(targets)