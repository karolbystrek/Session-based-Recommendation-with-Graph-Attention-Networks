# Graph Report - Session-based-Recommendation-with-Graph-Attention-Networks  (2026-05-22)

## Corpus Check
- 2 files · ~73,789 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 30 nodes · 28 edges · 6 communities (5 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5c0445e0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]

## God Nodes (most connected - your core abstractions)
1. `Study Description` - 16 edges
2. `Proposed Architectures` - 4 edges
3. `Model Comparison Plan` - 4 edges
4. `Main Hypotheses` - 4 edges
5. `Reference Architectures` - 3 edges
6. `Session-based Recommendation with Graph Attention Networks` - 1 edges
7. `Title` - 1 edges
8. `Overview` - 1 edges
9. `Motivation` - 1 edges
10. `Central Research Question` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities (6 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.15
Nodes (12): Central Research Question, Data and Preprocessing, Evaluation Protocol, Expected Contributions, Experimental Logic, Motivation, Overview, Session Graph Construction (+4 more)

### Community 1 - "Community 1"
Cohesion: 0.50
Nodes (4): H1: GAT improves graph encoding quality, H2: The effect of GAT depends on the readout mechanism, H3: A fully attention-based architecture may be especially effective, Main Hypotheses

### Community 2 - "Community 2"
Cohesion: 0.50
Nodes (4): Model Comparison Plan, Optional supplementary baselines, Original reference baselines, Reimplemented comparison models

### Community 3 - "Community 3"
Cohesion: 0.50
Nodes (4): Branch A: GAT-SR-GNN, Branch B: GAT-TAGNN, Branch C: GAT + SAGPool, Proposed Architectures

### Community 4 - "Community 4"
Cohesion: 0.67
Nodes (3): 1. [SR-GNN-style branch](./Session-based%20Recommendation%20with%20Graph%20Neural%20Networks.pdf), 2. [TAGNN-style branch](./TAGNN:%20Target%20Attentive%20Graph%20Neural%20Networks%20for%20Session-based%20Recommendation.pdf), Reference Architectures

## Knowledge Gaps
- **23 isolated node(s):** `Session-based Recommendation with Graph Attention Networks`, `Title`, `Overview`, `Motivation`, `Central Research Question` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Study Description` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.813) - this node is a cross-community bridge._
- **Why does `Proposed Architectures` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Why does `Model Comparison Plan` connect `Community 2` to `Community 0`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **What connects `Session-based Recommendation with Graph Attention Networks`, `Title`, `Overview` to the rest of the system?**
  _23 weakly-connected nodes found - possible documentation gaps or missing edges._