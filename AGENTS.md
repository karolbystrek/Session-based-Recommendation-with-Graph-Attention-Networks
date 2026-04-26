# AGENTS.md

## Repo Mission

Study: session-based next-item recommendation with graph attention.

Core question: does replacing GGNN encoder in SR-GNN/TAGNN-style graph session recommenders with GAT improve Precision@20 / MRR@20?

Main comparison:
- GGNN-style graph encoding vs GAT graph encoding.
- SR-GNN hybrid readout vs TAGNN target-aware readout vs SAGPool readout.
- Same datasets, preprocessing, evaluation protocol where possible.

## Research Background

Session = anonymous ordered click sequence. Task = predict next item.

Convert each session prefix to directed item-transition graph:
- node = unique item in prefix
- edge = consecutive click transition
- repeated transitions may be weighted / normalized
- preserve last-click node for local interest

Reference models:
- SR-GNN: GGNN over session graph, then local last-item embedding + global attention over session nodes.
- TAGNN: GGNN over session graph, then target-aware attention; session representation depends on candidate next item.

This repo investigates GAT replacement:
- keep session graph construction stable
- keep downstream readout stable for direct branches
- change one major component at a time
- measure if attention-based message passing improves node/session representations

## Proposed Models

`GAT-SR-GNN`
- Encoder: bidirectional GAT replaces GGNN.
- Readout: SR-GNN-style local + global attention session representation.
- Purpose: isolate encoder substitution under SR-GNN readout.
- Main notebook: `gat_sr_gnn_architecture.ipynb`.

`GAT-TAGNN`
- Encoder: bidirectional GAT replaces GGNN.
- Readout: TAGNN-style target-aware attention.
- Purpose: test if GAT helps with candidate-conditioned session scoring.
- Main notebook: `gat_tagnn_architecture.ipynb`.

`GAT-SAGPool`
- Encoder: GAT.
- Readout: SAGPool / self-attention graph pooling.
- Purpose: fully attention-based architecture.
- Current notebook: `gat_sagpool_architecture.ipynb`; preprocessing exists, model/training may still need completion.

## Current Repo Layout

- `README.md`: title only.
- `docs/study-description.md`: full research motivation, hypotheses, architecture rationale.
- `docs/study-roadmap.md`: implementation/evaluation plan.
- `docs/*.pdf`: reference papers: SR-GNN, TAGNN, GAT.
- `gat_sr_gnn_architecture.ipynb`: Kaggle-ready GAT + SR-GNN-style implementation.
- `gat_tagnn_architecture.ipynb`: Kaggle-ready GAT + TAGNN-style implementation.
- `gat_sagpool_architecture.ipynb`: Kaggle-ready preprocessing scaffold for GAT + SAGPool.
- `results/`: experiment outputs; should contain per-model histories, final metric tables, plots, and checkpoints.

## Notebook Search Hints

Use `rg` first.

Common symbols:
- preprocessing: `load_yoochoose_sessions`, `load_diginetica_sessions`, `drop_short_sessions`, `drop_rare_items`, `split_by_date`, `renumber_training_items`, `expand_sessions`, `preprocess`
- graph build: `build_session_graph`, `SessionGraphDataset`
- GAT encoder: `DirectionalGATStack`, `BidirectionalGATEncoder`
- SR-GNN model: `GATSRGNN`
- TAGNN model: `GATTAGNN`
- metrics/training: `precision_mrr_at_k`, `evaluate`, `train_one_epoch`, `run_dataset_experiment`

Likely useful searches:
- `rg -n "class |def |TODO|SAGPool|GATSRGNN|GATTAGNN|precision_mrr" *.ipynb`
- `rg -n "Yoochoose|Diginetica|Precision@20|MRR@20" .`

## Data Protocol

Datasets:
- Yoochoose
- Diginetica
- Yoochoose 1/64 used in implemented notebooks.
- Yoochoose 1/4 planned but skipped in notebooks due memory budget.

Preprocessing target:
- remove sessions length < 2
- remove items with frequency < 5
- temporal train/test split
- map train items to compact IDs
- remap test sessions to known train items
- expand session `[v1, v2, ..., vn]` to prefix-target examples:
  - `[v1] -> v2`
  - `[v1, v2] -> v3`
  - `[v1, ..., v(n-1)] -> vn`

Graph construction:
- lazy PyG graph creation in dataset
- one node per unique item in prefix
- directed forward and backward transition edges
- normalized edge weights where implemented
- keep mapping from sequence positions to local graph node indices

## Training / Evaluation Protocol

Primary metrics:
- Precision@20
- MRR@20

Optional metrics:
- HitRate@20
- Recall@20
- NDCG@20

Reference-like hyperparameters used in notebooks:
- hidden size `100`
- validation split `10%`
- Adam learning rate `0.001`
- LR decay factor `0.1` every `3` epochs
- batch size `100`
- L2 / weight decay `1e-5`
- early stopping patience `5`
- GAT-specific defaults: one bidirectional GAT layer, four heads, dropout `0.1`

Fair comparison rules:
- same preprocessing for all models
- same candidate space
- same validation/test protocol
- same tuning budget where feasible
- report best validation metrics + final test metrics
- use multiple seeds when experiment time allows

## Implementation Plan

Preferred order:
1. Verify preprocessing + graph builder.
2. Keep GAT-SR-GNN stable; use as working template.
3. Complete/run GAT-TAGNN; compare against TAGNN paper values.
4. Complete GAT-SAGPool model + training loop.
5. Add/reproduce SR-GNN and TAGNN GGNN baselines if needed for direct GGNN-vs-GAT claims.
6. Add ablations after core models work:
   - local only
   - average pooling
   - global attention
   - target-aware attention
   - SAGPool
   - GGNN encoder vs GAT encoder
7. Add multi-seed runs and significance/robustness checks.
8. Add qualitative analysis:
   - GAT attention weights
   - TAGNN target-aware attention shifts
   - SAGPool retained/suppressed nodes

## Coding Guidance For Agents

Repo currently notebook-centric. Preserve Kaggle-ready notebook flow unless user asks for package refactor.

When editing notebooks:
- avoid unrelated cell churn
- keep preprocessing consistent across notebooks
- keep model comparisons controlled
- update result paths/checkpoint names if adding model variants
- prefer reusable code patterns already present in `gat_sr_gnn_architecture.ipynb`

When implementing new model branches:
- do not change encoder and readout simultaneously unless branch requires it
- match tensor shapes expected by existing loaders and metrics
- mask padding item before ranking metrics
- ensure labels/item IDs align after remapping
- save history, final results, and checkpoint paths consistently

When reporting research claims:
- distinguish implemented result from planned experiment
- do not claim GAT outperforms GGNN without direct baseline under same protocol
- cite Precision@20 and MRR@20 with dataset and split
- mention memory constraint for Yoochoose 1/4 if skipped
