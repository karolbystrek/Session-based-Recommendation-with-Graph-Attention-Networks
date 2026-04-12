# Study Roadmap: GAT-Based Session Recommendation Benchmark

## 1. Define the study scope and research questions

- State the main goal:
  - Measure how replacing the GGNN encoder with a GAT encoder affects session-based recommendation performance.
- Define the comparison setting:
  - Keep the original second-stage designs from prior studies where possible.
  - Add a third, fully attention-based branch using SAGPool.
- Formulate the main research questions:
  - RQ1: Does replacing GGNN with GAT improve next-item recommendation quality?
  - RQ2: Does the effect of GAT depend on the downstream session-representation mechanism?
  - RQ3: Can a fully attention-based architecture with GAT + SAGPool outperform the adapted baselines?
- Decide the primary evaluation metrics:
  - Precision@20
  - MRR@20

---

## 2. Set up the experiment environment

- Create a dedicated project repository.
- Organize the repository into:
  - `data/`
  - `preprocessing/`
  - `models/`
  - `training/`
  - `evaluation/`
  - `configs/`
  - `results/`
  - `reports/`
- Fix reproducibility settings:
  - random seeds
  - package versions
  - hardware notes
  - logging format
- Choose the implementation stack:
  - Python
  - PyTorch
  - PyTorch Geometric or DGL for graph operations
- Create a configuration system for:
  - dataset paths
  - preprocessing switches
  - model hyperparameters
  - training hyperparameters
  - evaluation settings

---

## 3. Gather the datasets

- Download the same benchmark datasets used in the prior studies:
  - Yoochoose
  - Diginetica
- Store the raw files unchanged in `data/raw/`.
- Record:
  - source URL
  - download date
  - raw file names
  - raw dataset sizesyoo
- Verify that the dataset fields needed for session reconstruction exist:
  - session ID
  - item ID
  - timestamp
  - any other relevant metadata

---

## 4. Reproduce the preprocessing pipeline from the prior studies

- Reconstruct sessions ordered by timestamp.
- Apply the same filtering rules:
  - remove sessions with length < 2
  - remove items appearing fewer than 5 times
- Re-run filtering if needed until the data is consistent after item/session removal.
- Split data temporally, matching the prior work as closely as possible:
  - Yoochoose: most recent period used for test
  - Diginetica: most recent period used for test
- Save the exact split logic in code and in your experiment notes.
- Convert each session into prefix-label training examples:
  - `[v1] -> v2`
  - `[v1, v2] -> v3`
  - ...
  - `[v1, ..., v(n-1)] -> vn`
- For Yoochoose, also prepare the reduced subsets used in prior work:
  - Yoochoose 1/64
  - Yoochoose 1/4
- Save all processed artifacts:
  - item vocabulary
  - train sessions
  - validation sessions
  - test sessions
  - prefix-label examples
  - statistics summary

### Deliverables

- `data/processed/yoochoose_1_64/`
- `data/processed/yoochoose_1_4/`
- `data/processed/diginetica/`
- `reports/preprocessing_report.md`

---

## 5. Build a preprocessing validation checklist

- Confirm that:
  - the number of sessions matches expectations
  - the number of unique items is sensible
  - sessions are correctly ordered
  - no unknown target items leak into training improperly
  - prefix-target pairs are generated correctly
- Create sanity plots:
  - session length distribution
  - item frequency distribution
  - train/validation/test split counts
- Compare your processed dataset statistics with the papers wherever possible.

---

## 6. Define the common graph construction layer

- Implement a reusable session-to-graph converter.
- For each session:
  - create a directed graph
  - create one node per unique item in the session
  - add directed edges for consecutive clicks
  - assign edge weights according to the original session-graph construction
- Support repeated items within a session.
- Store:
  - node features / item IDs
  - adjacency structure
  - edge weights
  - mapping from local node indices to global item IDs
  - the last clicked item index
- Make the graph builder reusable across all model variants.

### Output

- One graph object per session-prefix example.

---

## 7. Implement the shared GAT encoder

- Replace the GGNN encoder with a GAT encoder.
- Design decisions to explicitly control:
  - number of GAT layers
  - number of attention heads
  - hidden dimension
  - dropout
  - residual connections
  - whether to concatenate or average heads
- Ensure the output format matches what the downstream modules expect:
  - one embedding vector per node/item in the session graph
- Keep the rest of the pipeline unchanged during the first adaptation pass.

### Important rule

- Change one major component at a time.
- First replace only the graph encoder, then test.
- Do not change both encoder and session readout simultaneously in the first round.

---

## 8. Implement the three main study architectures

### Model 1: GAT-SR-GNN

- Encoder:
  - GAT instead of GGNN
- Second phase:
  - SR-GNN-style hybrid session representation
  - local embedding from last item
  - global attention over session items
- Purpose:
  - isolate the effect of replacing GGNN with GAT in the SR-GNN framework

### Model 2: GAT-TAGNN

- Encoder:
  - GAT instead of GGNN
- Second phase:
  - TAGNN-style target-aware attention
  - local + global + target-conditioned representation
- Purpose:
  - test whether GAT helps when the downstream model already uses target-aware attention

### Model 3: GAT-SAGPool

- Encoder:
  - GAT
- Second phase:
  - SAGPool-based hierarchical graph pooling
  - fully attention-based end-to-end graph representation
- Purpose:
  - test a more cohesive attention-driven architecture instead of the original second phases

---

## 9. Add ablation variants

Create controlled variants so you can explain where gains come from.

### Encoder ablations

- GGNN encoder + original second phase
- GAT encoder + same second phase

### Readout ablations

- local only
- global average pooling
- global attention
- local + global attention
- target-aware attention
- GAT + SAGPool

### Graph construction ablations

- standard session graph
- optional higher-order or alternative connection variants
- only do these if time allows, after the core study is complete

---

## 10. Create a unified training pipeline

- Build one training script that accepts:
  - dataset
  - model type
  - encoder type
  - readout type
  - hyperparameters
  - output directory
- Standardize:
  - loss function
  - optimizer
  - learning rate schedule
  - early stopping rule
  - checkpoint saving
  - evaluation frequency
- Track:
  - training loss
  - validation Precision@20
  - validation MRR@20
  - epoch time
  - GPU memory if relevant
- Save:
  - best checkpoint by validation MRR@20
  - final checkpoint
  - full training log

---

## 11. Create a fair hyperparameter protocol

- Use a single tuning policy across all comparable models.
- Choose a validation split from the training data.
- Tune only a defined set of parameters:
  - hidden size
  - learning rate
  - dropout
  - number of layers
  - number of attention heads
  - weight decay
  - batch size
- Keep search budgets similar across models.
- Record every tried configuration.
- After choosing the best validation configuration, retrain on train + validation if that is part of your protocol.

### Recommendation

- Start from paper-like defaults where possible.
- Then tune minimally but consistently.

---

## 12. Run the full experiment matrix

For each dataset:

- Yoochoose 1/64
- Yoochoose 1/4
- Diginetica

For each model:

- original-style SR-GNN baseline
- original-style TAGNN baseline
- GAT-SR-GNN
- GAT-TAGNN
- GAT-SAGPool

For each configuration:

- run multiple seeds
- save mean and standard deviation

### Minimum recommended execution order

1. Preprocessing verification
2. SR-GNN baseline reproduction
3. TAGNN baseline reproduction
4. GAT-SR-GNN
5. GAT-TAGNN
6. GAT-SAGPool
7. Ablations
8. Error analysis

---

## 13. Evaluate the models

- Compute the main ranking metrics:
  - Precision@20
  - MRR@20
- Optionally also compute:
  - Hit Rate@20
  - Recall@20
  - NDCG@20
- Use the same candidate space and evaluation protocol for all models.
- Report:
  - best validation metrics
  - final test metrics
  - mean ± std over multiple seeds

---

## 14. Run statistical and robustness checks

- Compare:
  - GGNN vs GAT within the same architecture
  - SR-style second phase vs TAGNN-style second phase
  - both of the above vs GAT-SAGPool
- Run significance tests if feasible.
- Check robustness by:
  - short vs long sessions
  - frequent vs infrequent items
  - repeated-item sessions vs non-repeated sessions
- Inspect whether GAT helps more on:
  - longer sessions
  - noisier sessions
  - more complex transition structures

---

## 15. Perform qualitative analysis

- Visualize sample session graphs.
- Inspect learned attention weights for GAT-based models.
- For TAGNN-style models:
  - inspect how target-aware attention shifts for different candidate items
- For SAGPool:
  - inspect which nodes are retained or suppressed
- Build several case studies:
  - where GAT improves ranking
  - where GGNN remains better
  - where attention appears noisy or unstable

---

## 16. Summarize the results in publication-ready form

Create result tables for:

- baseline reproduction
- main comparison
- ablations
- robustness analyses

Create plots for:

- training curves
- Precision@20 across models
- MRR@20 across models
- performance by session length
- performance by dataset

Write down:

- what changed
- what stayed fixed
- why each comparison is fair

---

## 17. Draw final conclusions

- Answer whether GAT improves the graph encoding phase overall.
- Answer whether improvements transfer consistently across different second phases.
- Answer whether GAT + SAGPool provides a coherent end-to-end advantage.
- Identify:
  - best-performing model
  - most parameter-efficient model
  - most stable model across datasets

---

## 18. Package the study for reproducibility

- Release:
  - preprocessing scripts
  - config files
  - train/eval scripts
  - exact seeds
  - environment file
  - checkpoints if possible
- Add a `README` describing:
  - how to download data
  - how to preprocess
  - how to train each model
  - how to reproduce every table and figure

---

## 19. Final practical checklist

- [ ] Raw datasets downloaded
- [ ] Preprocessing reproduced
- [ ] Session graph builder validated
- [ ] SR-GNN baseline reproduced
- [ ] TAGNN baseline reproduced
- [ ] GAT encoder implemented
- [ ] GAT-SR-GNN trained
- [ ] GAT-TAGNN trained
- [ ] GAT-SAGPool trained
- [ ] Ablations completed
- [ ] Multi-seed evaluation completed
- [ ] Tables and figures generated
- [ ] Results interpreted
- [ ] Reproducibility package prepared
