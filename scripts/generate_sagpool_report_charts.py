"""Generate bar charts for GAT-SAGPOOL-RAPORT.html.

Run from repo root:
    python scripts/generate_sagpool_report_charts.py

Outputs PNG files to report-assets/sagpool/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "report-assets" / "sagpool"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Published paper baselines (same numbers as results_analysis.ipynb / notebooks).
PAPER = {
    "Yoochoose 1/64": {
        "SR-GNN (paper)": (70.570, 30.940),
        "TAGNN (paper)": (71.020, 31.120),
    },
    "Diginetica": {
        "SR-GNN (paper)": (50.730, 17.590),
        "TAGNN (paper)": (51.310, 18.030),
    },
}

# Selected direct GAT reimplementations (results-10).
GAT_BASELINES = {
    "Yoochoose 1/64": {
        "GAT-SR-GNN": (70.047, 31.097),
        "GAT-TAGNN": (70.888, 31.187),
    },
    "Diginetica": {
        "GAT-SR-GNN": (50.427, 16.688),
        "GAT-TAGNN": (51.362, 17.451),
    },
}

# SAGPool training phases (test metrics).
SAGPOOL_PHASES = {
    "Yoochoose 1/64": [
        ("Baseline SAGPool", 70.04, 30.90),
        ("Workstream A (cosine LR)", 69.02, 29.56),
        ("Phase 2B best (ratio=0.7)", 70.42, 31.16),
        ("Run 1 (v2)", 70.18, 31.01),
        ("Run 2 (digi reg)", 70.27, 31.08),
        ("Newest shared-edge GAT", 70.49, 31.16),
    ],
    "Diginetica": [
        ("Baseline SAGPool", 50.28, 16.66),
        ("Workstream A (cosine LR)", 47.10, 14.98),
        ("Workstream B", 50.36, 16.83),
        ("Run 1 (v2)", 50.47, 16.85),
        ("Run 2 (digi reg)", 48.77, 15.86),
        ("Newest shared-edge GAT", 48.93, 16.27),
    ],
}

PHASE2B = [
    ("b4_ratio07", 70.42, 31.16),
    ("b4_ratio04", 70.24, 31.16),
    ("b3_blocks2", 70.30, 31.08),
    ("b1_layers2", 70.10, 31.08),
    ("b7_gated", 70.31, 30.95),
    ("b5_gat_scorer", 70.16, 31.02),
    ("b2_heads8_h128", 69.97, 31.09),
    ("b6_hidden128", 69.80, 30.99),
]

PHASE3C = [
    ("c1_dropout03", 50.10, 16.59),
    ("c5_edge02", 50.04, 16.56),
    ("c2_emb01", 49.90, 16.65),
    ("c1_dropout02", 49.90, 16.36),
    ("c3_wd1e4", 49.18, 16.24),
]

COLORS = {
    "paper": "#94a3b8",
    "gat": "#3b82f6",
    "sagpool": "#f59e0b",
    "best": "#22c55e",
    "worse": "#ef4444",
}


def _bar_chart(
    labels,
    p20,
    mrr,
    title,
    filename,
    ref_lines=None,
    colors=None,
    rotate=35,
):
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 5.5))
    c1 = colors or [COLORS["sagpool"]] * len(labels)
    bars1 = ax.bar(x - width / 2, p20, width, label="Precision@20", color=c1, edgecolor="white")
    bars2 = ax.bar(x + width / 2, mrr, width, label="MRR@20", color=c1, alpha=0.55, edgecolor="white")

    if ref_lines:
        for name, val, style in ref_lines:
            ax.axhline(val, linestyle=style, linewidth=1.2, alpha=0.75, label=name)

    ax.set_ylabel("Wynik (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(p20), max(mrr)) * 1.12)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def chart_three_models_vs_paper():
    for dataset in ("Yoochoose 1/64", "Diginetica"):
        labels = []
        p20, mrr = [], []
        colors = []

        for name, (p, m) in PAPER[dataset].items():
            labels.append(name)
            p20.append(p)
            mrr.append(m)
            colors.append(COLORS["paper"])

        for name, (p, m) in GAT_BASELINES[dataset].items():
            labels.append(name)
            p20.append(p)
            mrr.append(m)
            colors.append(COLORS["gat"])

        # Best SAGPool per dataset
        if dataset == "Yoochoose 1/64":
            labels.append("GAT-SAGPool (best)")
            p20.append(70.49)
            mrr.append(31.16)
        else:
            labels.append("GAT-SAGPool (best)")
            p20.append(50.47)
            mrr.append(16.85)
        colors.append(COLORS["best"])

        safe = dataset.lower().replace(" ", "_").replace("/", "")
        _bar_chart(
            labels, p20, mrr,
            f"Porównanie modeli — {dataset} (test)",
            f"compare_all_models_{safe}.png",
            colors=colors,
            rotate=25,
        )


def chart_sagpool_phases():
    for dataset, phases in SAGPOOL_PHASES.items():
        labels = [p[0] for p in phases]
        p20 = [p[1] for p in phases]
        mrr = [p[2] for p in phases]
        paper_p = PAPER[dataset]["TAGNN (paper)"][0]
        paper_m = PAPER[dataset]["TAGNN (paper)"][1]
        safe = dataset.lower().replace(" ", "_").replace("/", "")
        _bar_chart(
            labels, p20, mrr,
            f"Fazy treningu GAT-SAGPool — {dataset}",
            f"sagpool_phases_{safe}.png",
            ref_lines=[
                (f"TAGNN paper P@20 ({paper_p:.2f})", paper_p, "--"),
                (f"TAGNN paper MRR ({paper_m:.2f})", paper_m, ":"),
            ],
            rotate=30,
        )


def chart_phase2b():
    labels = [p[0] for p in PHASE2B]
    p20 = [p[1] for p in PHASE2B]
    mrr = [p[2] for p in PHASE2B]
    _bar_chart(
        labels, p20, mrr,
        "Phase 2B — sweep architektury (Yoochoose 1/64, test)",
        "phase2b_sweep_yoochoose.png",
        ref_lines=[
            ("Baseline SAGPool P@20", 70.04, "--"),
            ("TAGNN paper P@20", 71.02, "-."),
        ],
        rotate=40,
    )


def chart_phase3c():
    labels = [p[0] for p in PHASE3C]
    p20 = [p[1] for p in PHASE3C]
    mrr = [p[2] for p in PHASE3C]
    _bar_chart(
        labels, p20, mrr,
        "Phase 3C — regularyzacja (Diginetica, test)",
        "phase3c_sweep_diginetica.png",
        ref_lines=[
            ("Run 1 (v2) P@20", 50.47, "--"),
            ("TAGNN paper P@20", 51.31, "-."),
        ],
        rotate=25,
    )


def chart_run1_vs_run2():
    labels = ["Run 1 (v2)", "Run 2 (reg Diginetica)", "Newest shared-edge GAT"]
    for dataset, r1, r2, newest in [
        ("Yoochoose 1/64", (70.18, 31.01), (70.27, 31.08), (70.49, 31.16)),
        ("Diginetica", (50.47, 16.85), (48.77, 15.86), (48.93, 16.27)),
    ]:
        p20 = [r1[0], r2[0], newest[0]]
        mrr = [r1[1], r2[1], newest[1]]
        colors = [COLORS["best"], COLORS["worse"], COLORS["sagpool"]]
        safe = dataset.lower().replace(" ", "_").replace("/", "")
        _bar_chart(
            labels, p20, mrr,
            f"Run 1 vs Run 2 vs Newest — {dataset}",
            f"run1_vs_run2_vs_newest_{safe}.png",
            colors=colors,
            rotate=0,
        )
        _bar_chart(
            labels, p20, mrr,
            f"Run 1 vs Run 2 vs Newest — {dataset}",
            f"run1_vs_run2_{safe}.png",
            colors=colors,
            rotate=0,
        )


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    chart_three_models_vs_paper()
    chart_sagpool_phases()
    chart_phase2b()
    chart_phase3c()
    chart_run1_vs_run2()
    print(f"\nDone. Charts in {OUT_DIR}")


if __name__ == "__main__":
    main()
