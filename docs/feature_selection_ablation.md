# Supervised Router Feature Study (Final Thesis Record)

## Scope

This document records the full feature-engineering and ablation process for the supervised query router used in dynamic weighted RRF, including methodology changes, intermediate findings, significance checks, and the final model selection used in thesis code.

Primary objective:
- Decide whether the final production router should use a compact subset or the complete feature set.

Primary metric:
- Dynamic weighted RRF NDCG@10.

Evaluation setting:
- Within-dataset paired splits.

---

## Final Decision

Final production choice: keep the full feature model.

Reason:
- A compact variant (compact_plus_11) appeared better in a short run, but the higher-repeat paired statistical rerun did not show a significant macro improvement over full.
- Therefore, the conservative thesis decision is to retain the full feature set.

---

## Canonical Final Feature Set (20 Features)

These features are implemented in src/retrieve_and_evaluate.py and reused by retrieval/evaluation, XGBoost optimization, and analysis scripts through get_selected_feature_names().

1. cross_entropy
- Mean token surprisal of query terms under corpus unigram statistics.

2. agreement
- Overlap ratio between sparse and dense top-k document sets.

3. query_length
- Number of query tokens before stopword removal.

4. dense_confidence
- Dense normalized top-1 minus top-2 score margin.

5. sparse_confidence
- Sparse normalized top-1 minus top-2 score margin.

6. confidence_gap
- dense_confidence minus sparse_confidence.

7. top_dense_score
- Dense normalized top-1 score.

8. top_sparse_score
- Sparse normalized top-1 score.

9. top_score_gap
- top_dense_score minus top_sparse_score.

10. average_idf
- Mean smoothed IDF over query terms.

11. max_idf
- Maximum smoothed IDF over query terms.

12. idf_std
- Standard deviation of smoothed IDF values.

13. rare_term_ratio
- Fraction of query terms with IDF above average_idf + idf_std.

14. stopword_ratio
- Ratio of stopwords in tokenized query.

15. first_shared_doc_rank
- Earliest average rank position of shared documents between sparse and dense top-k lists. Uses k+1 when no shared docs exist.

16. spearman_topk
- Spearman-style rank agreement across shared docs in top-k.

17. dense_entropy_topk
- Entropy of nonnegative dense top-k normalized scores.

18. sparse_entropy_topk
- Entropy of nonnegative sparse top-k normalized scores.

19. dense_concentration_topk
- Dense top-1 mass concentration inside top-k score mass.

20. sparse_concentration_topk
- Sparse top-1 mass concentration inside top-k score mass.

Notes:
- overlap_at_3 is intentionally removed from the final thesis feature set.
- feature_stat_k controls the top-k statistics used by the topk features.

---

## Process Timeline and Why Each Step Was Done

### Step 1: Move to single-feature LOFO ablation

Why:
- Group ablation could hide interactions and could not isolate per-feature impact precisely.

What was implemented:
- Within-dataset only.
- Paired splits across baseline and ablated runs.
- Delta definition fixed as:
  delta = NDCG_without_feature - NDCG_full.

Output artifacts:
- lofo_per_dataset_delta.csv
- lofo_macro_delta.csv
- lofo_macro_delta_plot.png

### Step 2: Use LOFO sign to build a compact candidate

Why:
- Features with negative delta were considered beneficial (removal hurts).
- Features with positive delta were considered potentially harmful.

What was implemented:
- compact feature set = features with macro LOFO delta < 0.

### Step 3: Build an incremental ladder between compact and full

Why:
- Test whether omitted features still carry useful signal.
- Avoid all-or-nothing compact vs full comparison.

What was implemented:
- Compare 14 models total:
  1) dense_only baseline
  2) full_router
  3) compact_plus_0 to compact_plus_11
- Each compact_plus_i adds one omitted feature in order of least harmful LOFO macro delta.

Output artifacts:
- model_ladder_plan.csv
- model_ladder_per_dataset.csv
- model_ladder_macro.csv
- model_ladder_macro_delta_vs_full.png
- model_ladder_dataset_delta_vs_full.png

### Step 4: Add paired statistical testing (full vs compact_plus_11)

Why:
- Mean deltas alone can be unstable with few repeats.
- Thesis claims require significance and confidence intervals.

What was implemented:
- Paired t-test on repeat-level differences.
- Wilcoxon signed-rank as a nonparametric companion.
- Per-dataset and macro summaries.

Output artifacts:
- full_vs_plus11_per_repeat_per_dataset.csv
- full_vs_plus11_per_repeat_macro.csv
- full_vs_plus11_stat_tests.csv

---

## Key Results by Stage

### A) LOFO Macro Deltas (initial run)

Highlights:
- Most beneficial to keep (negative delta):
  - top_score_gap: -0.001505
  - first_shared_doc_rank: -0.000664
  - sparse_confidence: -0.000657
  - agreement: -0.000442
- Most harmful on macro (positive delta):
  - query_length: +0.001215
  - dense_concentration_topk: +0.001168
  - sparse_concentration_topk: +0.001026

Interpretation:
- Useful directional signal for constructing compact candidates, but not enough by itself to lock final production features.

### B) Ladder Comparison (5 repeats)

Macro summary:
- full_router: 0.388566
- compact_plus_0: 0.386077
- compact_plus_11: 0.389781

Initial interpretation:
- compact_plus_11 looked better than full by +0.001215 and suggested that dropping one harmful feature might help.

### C) Paired significance test (5 repeats)

Macro (plus11 - full):
- mean_diff = +0.001215
- p(two-sided) = 0.003336
- CI95 = [+0.000676, +0.001755]

Interpretation at that time:
- Positive short-run signal, but repeat count still small.

### D) Stability rerun with 20 repeats (final check)

Macro (plus11 - full):
- mean_diff = +0.000206
- t = +0.5165
- p(two-sided) = 0.611495
- CI95 = [-0.000630, +0.001043]

Dataset-level pattern:
- Mixed signs (arguana positive, scidocs negative, others near zero), indicating no robust aggregate gain.

Final interpretation:
- The earlier apparent gain was not stable under a larger repeat budget.

---

## Final Thesis Conclusion

The final router keeps the full 20-feature set.

Rationale:
- Compact variants can occasionally outperform in short runs, but the 20-repeat paired test found no significant macro improvement over full.
- Full is the safest and most defensible final choice for thesis reporting and reproducibility.

---

## Reproducibility Notes

Core scripts used:
- src/ablation_study.py
- src/compare_negative_delta_features.py
- src/statistical_analysis_full_vs_plus11.py
- src/retrieve_and_evaluate.py
- src/optimize_xgboost.py
- src/analyze_xgboost_router.py

Important setting:
- within_dataset_evaluation.n_repeats controls the repeat budget unless overridden by CLI in statistical_analysis_full_vs_plus11.py.

Recommended thesis reporting practice:
- Report mean differences, confidence intervals, and p-values.
- Treat short-run ablation rankings as exploratory unless confirmed by higher-repeat paired tests.
