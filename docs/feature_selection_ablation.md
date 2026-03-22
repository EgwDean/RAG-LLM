# Feature Selection via Ablation for Supervised Query Routing

## Purpose

This ablation stage evaluates feature importance for supervised query routing in dynamic weighted Reciprocal Rank Fusion (wRRF). The objective is to identify a compact feature subset that preserves retrieval effectiveness while improving model simplicity and interpretability.

## Feature Groups and Features

### overlap
- `agreement`: overlap ratio between sparse and dense top-k result sets.
- `overlap_at_3`: overlap ratio between sparse and dense top-3 result sets.

### confidence
- `dense_confidence`: normalized top-1 minus top-2 dense score margin.
- `sparse_confidence`: normalized top-1 minus top-2 sparse score margin.
- `confidence_gap`: dense confidence minus sparse confidence.

### idf
- `average_idf`: average smoothed IDF of query terms.
- `max_idf`: maximum smoothed IDF among query terms.
- `idf_std`: standard deviation of smoothed IDF values.
- `rare_term_ratio`: proportion of query terms above a rarity threshold.

### query
- `query_length`: tokenized query length before stopword removal.
- `stopword_ratio`: ratio of stopwords in the tokenized query.

### entropy
- `cross_entropy`: query surprisal under corpus unigram statistics.

### legacy_topscore
- Score-based signal derived from top-ranked documents, using normalized top sparse/dense scores and their gap:
  - `top_dense_score`
  - `top_sparse_score`
  - `top_score_gap`

## Ablation Methodology

The study includes two complementary procedures:

1. Group ablation:
- Leave-one-out group ablation (remove one group at a time).
- Single-group ablation (use one group at a time).

2. Candidate feature-set comparison:
- `FULL`
- `CORE`
- `CORE_PLUS_QUERY`

Primary metric:
- Dynamic weighted RRF performance measured by NDCG@k (`dynamic_wrrf_ndcg`).

Evaluation mode:
- Within-dataset evaluation.

## Candidate Feature-Set Results

- `FULL` (15 features): **0.380782**
- `CORE` (8 features): **0.379642**
- `CORE_PLUS_QUERY` (10 features): **0.380555**

## Interpretation

- `FULL` achieves the highest score, but only marginally better than `CORE_PLUS_QUERY`.
- `CORE_PLUS_QUERY` achieves nearly identical performance with fewer features.
- `CORE` alone shows slight degradation, indicating that query features add useful information.
- Additional `idf` and `entropy` features contribute minimally in this setting.

## Final Conclusion

Selected feature set: **CORE_PLUS_QUERY**.

Justification:
- Best trade-off between effectiveness and simplicity.
- Lower dimensionality than `FULL`.
- Improved interpretability through focused feature groups.
- Reduced risk of overfitting from weak or marginal features.
