from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results-analysis" / "all_model_runs.csv"
OUT_DIR = ROOT / "presentation-assets"

DATASETS = ["Yoochoose 1/64", "Diginetica"]
MODEL_ORDER = ["SR-GNN paper", "TAGNN paper", "GAT-SR-GNN", "GAT-TAGNN", "GAT-SAGPool"]
MODEL_COLORS = {
    "SR-GNN paper": "#6f7782",
    "TAGNN paper": "#9aa0a6",
    "GAT-SR-GNN": "#1f77b4",
    "GAT-TAGNN": "#2ca02c",
    "GAT-SAGPool": "#d95f02",
}

PAPER_BASELINES = {
    ("SR-GNN paper", "Yoochoose 1/64"): {"test_precision@20": 70.57, "test_mrr@20": 30.94},
    ("SR-GNN paper", "Diginetica"): {"test_precision@20": 50.73, "test_mrr@20": 17.59},
    ("TAGNN paper", "Yoochoose 1/64"): {"test_precision@20": 71.02, "test_mrr@20": 31.12},
    ("TAGNN paper", "Diginetica"): {"test_precision@20": 51.31, "test_mrr@20": 18.03},
}

FINAL_SELECTION = {
    "GAT-SR-GNN": 10,
    "GAT-TAGNN": 10,
    "GAT-SAGPool": 6,  # results-newest: final shared-edge GAT variant
}


def load_runs() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS)
    runs = runs.dropna(subset=["test_precision@20", "test_mrr@20"])
    runs["iteration"] = runs["iteration"].astype(int)
    return runs


def selected_results(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, dataset), metrics in PAPER_BASELINES.items():
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "test_precision@20": metrics["test_precision@20"],
                "test_mrr@20": metrics["test_mrr@20"],
            }
        )

    for model, iteration in FINAL_SELECTION.items():
        selected = runs[(runs["model"] == model) & (runs["iteration"] == iteration)]
        for dataset in DATASETS:
            row = selected[selected["dataset"] == dataset].iloc[0]
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "test_precision@20": row["test_precision@20"],
                    "test_mrr@20": row["test_mrr@20"],
                }
            )

    return pd.DataFrame(rows)


def style_axes(ax):
    ax.grid(axis="y", color="#d8dde6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa0a6")
    ax.spines["bottom"].set_color("#9aa0a6")


def padded_axis_limits(values, step):
    lower = values.min()
    upper = values.max()
    padding = max(step, (upper - lower) * 0.18)
    y_min = max(0, math.floor((lower - padding) / step) * step)
    y_max = math.ceil((upper + padding) / step) * step
    return y_min, y_max


def plot_main_comparison(data: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    metrics = [
        ("test_precision@20", "Precision@20"),
        ("test_mrr@20", "MRR@20"),
    ]

    for row_idx, dataset in enumerate(DATASETS):
        dataset_data = data[data["dataset"] == dataset].set_index("model").loc[MODEL_ORDER]
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            values = dataset_data[metric]
            colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
            bars = ax.bar(MODEL_ORDER, values, color=colors, width=0.68)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
            ax.set_title(f"{dataset} - {title}", fontsize=13, weight="bold")
            ax.set_ylabel(title)
            ax.tick_params(axis="x", labelrotation=24, labelsize=9)
            ax.set_ylim(0, 100 if metric == "test_precision@20" else 50)
            style_axes(ax)

    fig.savefig(OUT_DIR / "main_results_2x2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_iteration_progression(runs: pd.DataFrame, metric: str, label: str, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), sharey=False, constrained_layout=True)
    model_styles = {
        "GAT-SR-GNN": {"color": "#1f77b4", "marker": "o"},
        "GAT-TAGNN": {"color": "#2ca02c", "marker": "s"},
        "GAT-SAGPool": {"color": "#d95f02", "marker": "^"},
    }

    for ax, dataset in zip(axes, DATASETS):
        dataset_runs = runs[runs["dataset"] == dataset]
        for model, style in model_styles.items():
            model_runs = dataset_runs[dataset_runs["model"] == model].sort_values("iteration")
            ax.plot(
                model_runs["iteration"],
                model_runs[metric],
                linewidth=2.2,
                markersize=5.5,
                label=model,
                **style,
            )
        ax.set_title(dataset, fontsize=13, weight="bold")
        ax.set_xlabel("Iteration / archived run")
        ax.set_ylabel(label)
        ax.set_xticks(sorted(dataset_runs["iteration"].unique()))
        step = 1.0 if metric == "test_precision@20" else 0.5
        ax.set_ylim(*padded_axis_limits(dataset_runs[metric], step))
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Model Progression by {label}", fontsize=18, weight="bold", y=1.13)
    fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_deltas(data: pd.DataFrame):
    final_gat = data[data["model"].str.startswith("GAT-")].copy()
    baseline = data[data["model"].isin(["SR-GNN paper", "TAGNN paper"])].copy()

    rows = []
    for _, row in final_gat.iterrows():
        if row["model"] == "GAT-TAGNN":
            reference = "TAGNN paper"
        else:
            reference = "SR-GNN paper"
        ref = baseline[(baseline["model"] == reference) & (baseline["dataset"] == row["dataset"])].iloc[0]
        rows.append(
            {
                "label": f"{row['model'].replace('GAT-', '')}\n{row['dataset']}",
                "precision_delta": row["test_precision@20"] - ref["test_precision@20"],
                "mrr_delta": row["test_mrr@20"] - ref["test_mrr@20"],
            }
        )

    delta = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "precision_delta", "Precision@20 delta"),
        (axes[1], "mrr_delta", "MRR@20 delta"),
    ]:
        colors = ["#2ca02c" if value >= 0 else "#c44e52" for value in delta[metric]]
        bars = ax.bar(delta["label"], delta[metric], color=colors, width=0.68)
        ax.axhline(0, color="#202124", linewidth=1)
        ax.bar_label(bars, fmt="%+.2f", padding=3, fontsize=8)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_ylabel("Absolute delta")
        ax.tick_params(axis="x", labelrotation=20, labelsize=9)
        style_axes(ax)

    fig.suptitle("Final GAT Architectures vs Matching Paper Baseline", fontsize=17, weight="bold")
    fig.savefig(OUT_DIR / "final_deltas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    runs = load_runs()
    final = selected_results(runs)
    final.to_csv(OUT_DIR / "presentation_final_results.csv", index=False)

    plot_main_comparison(final)
    plot_iteration_progression(runs, "test_precision@20", "Precision@20", "precision_progression.png")
    plot_iteration_progression(runs, "test_mrr@20", "MRR@20", "mrr_progression.png")
    plot_final_deltas(final)


if __name__ == "__main__":
    main()
