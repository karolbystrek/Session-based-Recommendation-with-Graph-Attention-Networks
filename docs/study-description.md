# Study Description

## Title

Investigating Graph Attention Networks as a Replacement for GGNN Encoders in Graph-Based Session Recommendation

## Overview

This study examines whether replacing the **Gated Graph Neural Network (GGNN)** encoder used in established graph-based session recommender systems with a **Graph Attention Network (GAT)** can improve next-item prediction performance in anonymous session-based recommendation. The work is anchored in two influential reference models, **SR-GNN** and **TAGNN**, both of which represent sessions as **directed item-transition graphs** in order to model structural relationships between clicked items that are difficult to capture with purely sequential approaches.

The core idea of the study is to keep the overall logic of these reference architectures as intact as possible while **systematically substituting their shared first-stage graph encoder**. In the original models, this stage is based on GGNN-style gated message passing. In the proposed study, it will be replaced with a GAT-based encoder so that the effect of **attention-driven neighborhood aggregation** can be evaluated under controlled conditions.

## Motivation

Graph-based session recommenders were introduced to address a central limitation of earlier sequential methods: they often model only local, consecutive transitions and therefore miss more complex relationships between items inside a session. SR-GNN showed that representing each session as a graph allows the model to capture richer item-transition structure and then summarize the session through a combination of **global preference** and **current interest**. TAGNN extended this idea by introducing **target-aware attention**, allowing the session representation to vary depending on the candidate next item being scored.

Although both architectures differ in how they construct the final session representation, they still rely on the same general graph-learning principle in the first phase: **GGNN-based propagation over the session graph**. This creates a clear experimental opportunity. If GGNN is replaced by GAT, the study can directly test whether **attention-based message passing** produces better node embeddings, better session representations, and better recommendation accuracy than recurrent gated propagation.

## Central Research Question

The central question of the study is:

**Does replacing the GGNN-based graph encoder in SR-GNN- and TAGNN-style session recommenders with a GAT encoder improve session-based recommendation performance, and how do any gains depend on the downstream session-representation mechanism?**

This main question can be divided into three subordinate questions:

1. **Encoder-level question:**  
   Does GAT learn stronger item representations than GGNN when sessions are modeled as directed graphs?

2. **Interaction question:**  
   Does the effect of GAT depend on the type of second-stage session modeling used after graph encoding?

3. **Architecture question:**  
   Can a fully attention-based design, using GAT together with **SAGPool**, produce a more cohesive and more effective end-to-end recommender?

## Reference Architectures

The study is built around two baseline graph-based session recommendation frameworks:

### 1. [SR-GNN-style branch](./Session-based%20Recommendation%20with%20Graph%20Neural%20Networks.pdf)

In SR-GNN, each session is converted into a directed session graph and item embeddings are learned through graph propagation. The session representation is then formed as a **hybrid of two components**:

- a **global preference** representation summarizing the broader behavioral pattern in the session,
- a **current interest** representation emphasizing the most recent or most relevant short-term behavior.

This branch of the study will preserve that second-stage design while replacing only the GGNN encoder with GAT.

### 2. [TAGNN-style branch](./TAGNN:%20Target%20Attentive%20Graph%20Neural%20Networks%20for%20Session-based%20Recommendation.pdf)

TAGNN also uses graph-based item encoding, but differs in the readout stage. Instead of using a single fixed session vector, it introduces **target-aware attention**, so that the representation of the session is conditioned on the candidate next item. This makes the session embedding adaptive rather than static.

This branch of the study will preserve that target-aware second stage while again replacing the GGNN encoder with GAT.

## Proposed Architectures

The experimental design contains **three complementary model branches**.

### Branch A: GAT-SR-GNN

This model reproduces the SR-GNN pipeline as closely as possible, except that the GGNN-based graph encoder is replaced by a GAT encoder.

Its purpose is to test whether attention-based neighborhood aggregation improves performance when the downstream session representation is still based on the SR-GNN idea of combining:

- global session preference,
- current session interest,
- attention-based session summarization.

This branch isolates the effect of GAT under the SR-GNN readout mechanism.

### Branch B: GAT-TAGNN

This model reproduces the TAGNN pipeline while replacing its GGNN encoder with GAT.

Its purpose is to test whether GAT works particularly well when paired with a **target-aware second phase**, where the session representation changes depending on which candidate item is being scored. This branch is especially important because target-aware attention may amplify any gains produced by better node embeddings.

### Branch C: GAT + SAGPool

This branch introduces a third architecture that differs more substantially from the two direct reproductions. Instead of using the SR-GNN hybrid readout or TAGNN target-aware readout, it uses **SAGPool (Self-Attention Graph Pooling)** as the second-stage mechanism after GAT-based node encoding.

The objective of this branch is to construct a **fully attention-based end-to-end architecture**, where:

- node representations are learned through graph attention,
- graph-level/session-level summarization is performed through self-attention graph pooling,
- the overall model avoids recurrent gated propagation entirely.

This third branch makes the study not only comparative but also exploratory, testing whether a more internally consistent attention-based architecture can further improve performance.

## Experimental Logic

The study is designed as a **controlled comparative investigation** rather than a completely novel model proposal without anchors. The main experimental principle is:

- keep the session-graph construction consistent with the reference studies,
- keep the training/evaluation setup aligned with the original papers,
- alter the graph encoder in a controlled way,
- compare performance across multiple downstream readout mechanisms.

This design makes it possible to distinguish between three different sources of improvement:

1. improvement caused by **better graph propagation** alone,
2. improvement caused by the **interaction between GAT and a specific session readout mechanism**,
3. improvement caused by moving toward a **fully attention-based architecture**.

## Data and Preprocessing

For comparability with the reference works, the study will use the same benchmark datasets and the same general preprocessing pipeline:

- **Yoochoose**
- **Diginetica**

The preprocessing protocol will follow the reference studies as closely as possible:

1. remove items that appear fewer than **five times**,
2. remove sessions of length fewer than **two interactions**,
3. split the data **temporally** into training and test portions,
4. transform each session into multiple **prefix-target pairs** for next-item prediction.

For example, a session of the form:

`[v1, v2, v3, v4]`

is converted into training/evaluation examples such as:

- `([v1], v2)`
- `([v1, v2], v3)`
- `([v1, v2, v3], v4)`

This setup ensures that the task remains the standard **session-based next-item recommendation** problem used in the original studies.

## Session Graph Construction

Each session will be represented as a **directed session graph** in which:

- each node corresponds to an item appearing in the session,
- each directed edge represents a transition from one clicked item to the next,
- repeated transitions may be encoded through edge weighting or normalized adjacency, consistent with the original graph construction schemes.

This graph representation is important because it allows the model to capture not only immediate adjacency but also richer transition structure within the session.

## Model Comparison Plan

The study will compare at least the following categories of models:

### Original reference baselines

- SR-GNN
- TAGNN

### Reimplemented comparison models

- GAT-SR-GNN
- GAT-TAGNN
- GAT + SAGPool

### Optional supplementary baselines

If computationally feasible, the study may also include simpler or commonly cited baselines from the literature for additional context, but the primary focus remains the comparison between:

- **GGNN-based graph encoding**
- **GAT-based graph encoding**

under otherwise comparable settings.

## Evaluation Protocol

The evaluation will follow the standard metrics used in the reference studies:

- **Precision@20**
- **MRR@20**

These metrics are appropriate because the task is top-k next-item recommendation. Precision@20 measures whether the correct next item appears among the top recommended candidates, while MRR@20 also accounts for the rank position of the correct item within that top-20 list.

The evaluation goal is not merely to show whether one model wins overall, but to determine:

- whether GAT improves recommendation quality consistently,
- whether the improvements differ across architectures,
- whether the benefits are stronger in one downstream readout design than another.

## Main Hypotheses

The study is guided by the following hypotheses:

### H1: GAT improves graph encoding quality

Because GAT applies attention over graph neighborhoods, it may learn more selective and informative item representations than GGNN, especially in sessions where not all neighboring transitions are equally informative.

### H2: The effect of GAT depends on the readout mechanism

The advantage of better graph embeddings may not be uniform across all architectures. It may interact differently with:

- SR-GNN’s hybrid global/local representation,
- TAGNN’s target-aware attention,
- SAGPool’s attention-based graph pooling.

### H3: A fully attention-based architecture may be especially effective

Combining GAT with SAGPool may yield stronger representational coherence than hybrid architectures that mix attention-based propagation with non-attention-based aggregation, potentially improving end-to-end recommendation quality.

## Expected Contributions

The study is expected to make the following contributions:

1. **A controlled encoder substitution study**  
   It isolates the effect of replacing GGNN with GAT in two established graph-based session recommendation architectures.

2. **A cross-architecture comparison**  
   It evaluates whether the usefulness of GAT is architecture-dependent rather than universally beneficial.

3. **An extensible third branch with SAGPool**  
   It introduces a fully attention-based alternative that goes beyond direct reproduction and opens a path toward more unified graph-attention recommender designs.

4. **A reproducible benchmarking setup**  
   By using the same datasets, preprocessing procedures, and evaluation metrics as the reference studies, it provides a fair and interpretable comparison.

## Significance of the Study

This study is significant because it addresses a precise and meaningful methodological question in session-based recommendation: whether the gains of graph-based recommenders come mainly from the **graph formulation itself**, or whether additional improvements can be obtained by upgrading the graph encoder from recurrent gated propagation to **attention-based propagation**.

More broadly, the study contributes to understanding how architectural choices in graph neural recommender systems should be made. Rather than treating “graph neural network” as a single design category, it analyzes how different graph encoders and readout mechanisms interact, and whether a unified attention-based approach can produce stronger performance in next-item prediction.

## Summary

In summary, the proposed research revisits two influential graph-based session recommendation models, SR-GNN and TAGNN, and systematically replaces their shared GGNN encoder with a Graph Attention Network. It then evaluates this encoder substitution in three settings: an SR-GNN-style readout, a TAGNN-style target-aware readout, and a new fully attention-based GAT+SAGPool architecture. Using the same benchmark datasets, preprocessing strategy, and evaluation protocol as the original studies, the work aims to deliver a controlled and interpretable assessment of how attention-based graph propagation influences item representation learning, session modeling, and final recommendation accuracy in session-based recommendation.
