"""statistical_analysis_full_vs_plus11.py

Paired statistical analysis for full router vs compact_plus_11 router.

Method:
- Within-dataset paired splits (same split per repeat for both models)
- Compare dynamic WRRF NDCG@k
- Export per-repeat scores and statistical tests

Test definitions:
- diff = plus11_ndcg - full_ndcg
- Paired t-test on diffs (H0: mean(diff) = 0)
- Wilcoxon signed-rank test on diffs (two-sided)
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


# Ensure project root is importable and set as working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.retrieve_and_evaluate import (
    apply_zscore,
    build_or_load_query_feature_cache,
    compute_zscore_stats,
    dataset_seed_offset,
    ensure_retrieval_results_cached,
    evaluate_benchmark_methods_for_qids,
    get_selected_feature_names,
    predict_router_alpha,
    rows_to_matrix_with_features,
    set_global_seed,
    split_rows_train_test,
    train_router_model,
)
from src.utils import ensure_dir, get_config_path, load_config, model_short_name


def load_macro_delta_map(macro_delta_csv):
    """Load feature->macro_delta mapping from LOFO macro CSV."""
    delta_map = {}
    with open(macro_delta_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = str(row["feature_removed"]).strip()
            delta_map[feature] = float(row["macro_delta_ndcg"])
    return delta_map


def build_plus_model_features(full_features, macro_delta_map, delta_threshold, plus_index):
    """Construct compact_plus_{plus_index} feature set using LOFO ordering."""
    compact_features = [f for f in full_features if macro_delta_map.get(f, float("inf")) < delta_threshold]
    if not compact_features:
        raise ValueError("No compact features found. Adjust delta threshold or verify LOFO macro CSV.")

    omitted = [f for f in full_features if f not in compact_features]
    if not omitted:
        raise ValueError("No omitted features available; compact already equals full.")

    omitted_sorted = sorted(omitted, key=lambda f: (macro_delta_map.get(f, float("inf")), f))

    if plus_index < 0:
        raise ValueError("plus_index must be >= 0.")

    n_add = min(plus_index, len(omitted_sorted))
    added = set(omitted_sorted[:n_add])
    plus_features = [f for f in full_features if (f in compact_features or f in added)]

    return compact_features, omitted_sorted, plus_features


def run_model_once(train_rows, test_rows, feature_names, cfg, dataset_name, ds_cache, device, ndcg_k, rrf_k):
    """Train and evaluate one model on one paired split."""
    X_train_raw, y_train, _ = rows_to_matrix_with_features(train_rows, feature_names)
    X_test_raw, _, test_qids = rows_to_matrix_with_features(test_rows, feature_names)

    train_mean, train_std = compute_zscore_stats(X_train_raw)
    X_train = apply_zscore(X_train_raw, train_mean, train_std)
    X_test = apply_zscore(X_test_raw, train_mean, train_std)

    model_bundle = train_router_model(
        X_train,
        y_train,
        cfg,
        device,
        dataset_name=dataset_name,
        optimization_mode="within_dataset",
    )
    alphas = predict_router_alpha(model_bundle, X_test, cfg, device)
    alpha_map = {qid: float(alpha) for qid, alpha in zip(test_qids, alphas)}

    metrics = evaluate_benchmark_methods_for_qids(
        bm25_results=ds_cache["bm25_results"],
        dense_results=ds_cache["dense_results"],
        qrels=ds_cache["qrels"],
        ndcg_k=ndcg_k,
        rrf_k=rrf_k,
        query_ids=test_qids,
        alpha_map=alpha_map,
    )
    return float(metrics["dynamic_wrrf_ndcg"])


def summarize_differences(name, diffs):
    """Compute paired statistics for a difference vector."""
    diffs = np.asarray(diffs, dtype=np.float64)
    n = int(diffs.shape[0])
    if n < 2:
        raise ValueError(f"Need at least 2 paired samples for {name}; got {n}.")

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = float(std_diff / math.sqrt(n)) if std_diff > 0.0 else 0.0
    df = n - 1

    if std_diff <= 0.0:
        t_stat = 0.0
        p_two_sided = 1.0
        p_one_sided_plus_gt_full = 0.5
    else:
        t_stat = float(mean_diff / se_diff)
        if scipy_stats is None:
            p_two_sided = float("nan")
            p_one_sided_plus_gt_full = float("nan")
        else:
            p_two_sided = float(2.0 * scipy_stats.t.sf(abs(t_stat), df=df))
            p_one_sided_plus_gt_full = float(scipy_stats.t.sf(t_stat, df=df))

    if scipy_stats is None:
        ci_low = float("nan")
        ci_high = float("nan")
    elif std_diff <= 0.0:
        ci_low = mean_diff
        ci_high = mean_diff
    else:
        t_crit = float(scipy_stats.t.ppf(0.975, df=df))
        ci_low = float(mean_diff - t_crit * se_diff)
        ci_high = float(mean_diff + t_crit * se_diff)

    if std_diff <= 0.0:
        cohen_dz = 0.0
    else:
        cohen_dz = float(mean_diff / std_diff)

    wilcoxon_stat = float("nan")
    wilcoxon_p_two_sided = float("nan")
    if scipy_stats is not None:
        try:
            w = scipy_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat = float(w.statistic)
            wilcoxon_p_two_sided = float(w.pvalue)
        except ValueError:
            # All-zero differences or degenerate case.
            wilcoxon_stat = 0.0
            wilcoxon_p_two_sided = 1.0

    return {
        "name": name,
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "df": df,
        "t_stat": t_stat,
        "p_two_sided": p_two_sided,
        "p_one_sided_plus_gt_full": p_one_sided_plus_gt_full,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "cohen_dz": cohen_dz,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p_two_sided": wilcoxon_p_two_sided,
    }


def save_repeat_dataset_csv(rows, output_csv):
    """Write per-repeat per-dataset paired scores."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "repeat_index",
            "dataset",
            "full_ndcg",
            "plus_ndcg",
            "diff_plus_minus_full",
        ])
        for row in rows:
            writer.writerow([
                row["repeat_index"],
                row["dataset"],
                f"{row['full_ndcg']:.6f}",
                f"{row['plus_ndcg']:.6f}",
                f"{row['diff_plus_minus_full']:.6f}",
            ])


def save_repeat_macro_csv(rows, output_csv):
    """Write per-repeat macro paired scores across datasets."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "repeat_index",
            "macro_full_ndcg",
            "macro_plus_ndcg",
            "macro_diff_plus_minus_full",
        ])
        for row in rows:
            writer.writerow([
                row["repeat_index"],
                f"{row['macro_full_ndcg']:.6f}",
                f"{row['macro_plus_ndcg']:.6f}",
                f"{row['macro_diff_plus_minus_full']:.6f}",
            ])


def save_test_summary_csv(rows, output_csv):
    """Write statistical test summary rows."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scope",
            "n",
            "mean_diff",
            "std_diff",
            "se_diff",
            "df",
            "t_stat",
            "p_two_sided",
            "p_one_sided_plus_gt_full",
            "ci95_low",
            "ci95_high",
            "cohen_dz",
            "wilcoxon_stat",
            "wilcoxon_p_two_sided",
        ])
        for row in rows:
            writer.writerow([
                row["name"],
                row["n"],
                f"{row['mean_diff']:.6f}",
                f"{row['std_diff']:.6f}",
                f"{row['se_diff']:.6f}",
                row["df"],
                f"{row['t_stat']:.6f}",
                f"{row['p_two_sided']:.6g}",
                f"{row['p_one_sided_plus_gt_full']:.6g}",
                f"{row['ci95_low']:.6f}",
                f"{row['ci95_high']:.6f}",
                f"{row['cohen_dz']:.6f}",
                f"{row['wilcoxon_stat']:.6f}",
                f"{row['wilcoxon_p_two_sided']:.6g}",
            ])


def main():
    parser = argparse.ArgumentParser(
        description="Run paired statistical analysis for full router vs compact_plus_11."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--macro-delta-csv",
        default=None,
        help="Optional path to lofo_macro_delta.csv. Defaults to model results folder.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.0,
        help="Compact feature threshold: keep features where macro_delta_ndcg < threshold.",
    )
    parser.add_argument(
        "--plus-index",
        type=int,
        default=11,
        help="Use compact_plus_{plus_index} feature set (default: 11).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Override within_dataset_evaluation.n_repeats from config.",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets configured.")

    cfg.setdefault("supervised_routing", {})
    cfg["supervised_routing"]["model_type"] = "xgboost"
    routing_cfg = cfg.get("supervised_routing", {})

    seed = int(routing_cfg.get("seed", 42))
    set_global_seed(seed)

    use_cuda = bool(routing_cfg.get("use_cuda_if_available", True))
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    within_cfg = cfg.get("within_dataset_evaluation", {})
    train_fraction = float(within_cfg.get("train_fraction", 0.8))
    n_repeats = int(within_cfg.get("n_repeats", 5) if args.n_repeats is None else args.n_repeats)
    shuffle = bool(within_cfg.get("shuffle", True))

    if n_repeats <= 1:
        raise ValueError("n_repeats must be > 1 for statistical testing.")

    ndcg_k = int(cfg["benchmark"].get("ndcg_k", 10))
    rrf_k = int(cfg["benchmark"].get("rrf", {}).get("k", 60))
    short_model = model_short_name(cfg["embeddings"]["model_name"])

    results_root = get_config_path(cfg, "results_folder", "data/results")
    ablation_dir = os.path.join(results_root, short_model, "ablation")
    macro_delta_csv = args.macro_delta_csv or os.path.join(ablation_dir, "lofo_macro_delta.csv")
    if not os.path.exists(macro_delta_csv):
        raise FileNotFoundError(f"Macro delta CSV not found: {macro_delta_csv}")

    full_features = get_selected_feature_names()
    macro_delta_map = load_macro_delta_map(macro_delta_csv)
    compact_features, omitted_sorted, plus_features = build_plus_model_features(
        full_features=full_features,
        macro_delta_map=macro_delta_map,
        delta_threshold=args.delta_threshold,
        plus_index=args.plus_index,
    )

    print("=" * 72)
    print("Statistical analysis: full router vs compact_plus model")
    print(f"Device                 : {device}")
    print(f"Datasets ({len(datasets)})         : {', '.join(datasets)}")
    print(f"n_repeats              : {n_repeats}")
    print(f"train_fraction         : {train_fraction}")
    print(f"full feature count     : {len(full_features)}")
    print(f"compact feature count  : {len(compact_features)}")
    print(f"plus_index             : {args.plus_index}")
    print(f"plus feature count     : {len(plus_features)}")
    print(f"macro delta CSV        : {macro_delta_csv}")
    print("diff definition        : plus_ndcg - full_ndcg")
    if omitted_sorted:
        n_show = min(args.plus_index, len(omitted_sorted))
        if n_show > 0:
            print(f"added features         : {omitted_sorted[:n_show]}")
    print("=" * 72)

    print("\n[1/4] Loading cached retrieval artifacts per dataset ...")
    dataset_cache_map = {}
    for dataset_name in datasets:
        dataset_cache_map[dataset_name] = ensure_retrieval_results_cached(dataset_name, cfg, device)

    print("\n[2/4] Loading or building query feature cache ...")
    all_rows, _, _ = build_or_load_query_feature_cache(dataset_cache_map, cfg, short_model)
    rows_by_dataset = {ds: [] for ds in datasets}
    for row in all_rows:
        rows_by_dataset[row["dataset"]].append(row)
    for ds in datasets:
        rows_by_dataset[ds].sort(key=lambda r: r["query_id"])

    print("\n[3/4] Running paired repeats for full vs plus model ...")
    per_repeat_dataset_rows = []
    per_repeat_macro_rows = []

    for repeat_idx in range(n_repeats):
        full_vals = []
        plus_vals = []

        for dataset_name in datasets:
            ds_rows = list(rows_by_dataset[dataset_name])
            if len(ds_rows) < 2:
                raise ValueError(f"Dataset {dataset_name} has only {len(ds_rows)} rows; need at least 2.")

            repeat_seed = seed + repeat_idx + dataset_seed_offset(dataset_name)
            train_rows, test_rows = split_rows_train_test(
                ds_rows,
                train_fraction=train_fraction,
                repeat_seed=repeat_seed,
                shuffle=shuffle,
            )

            full_ndcg = run_model_once(
                train_rows=train_rows,
                test_rows=test_rows,
                feature_names=full_features,
                cfg=cfg,
                dataset_name=dataset_name,
                ds_cache=dataset_cache_map[dataset_name],
                device=device,
                ndcg_k=ndcg_k,
                rrf_k=rrf_k,
            )
            plus_ndcg = run_model_once(
                train_rows=train_rows,
                test_rows=test_rows,
                feature_names=plus_features,
                cfg=cfg,
                dataset_name=dataset_name,
                ds_cache=dataset_cache_map[dataset_name],
                device=device,
                ndcg_k=ndcg_k,
                rrf_k=rrf_k,
            )

            diff = float(plus_ndcg - full_ndcg)
            per_repeat_dataset_rows.append(
                {
                    "repeat_index": repeat_idx,
                    "dataset": dataset_name,
                    "full_ndcg": float(full_ndcg),
                    "plus_ndcg": float(plus_ndcg),
                    "diff_plus_minus_full": diff,
                }
            )

            full_vals.append(float(full_ndcg))
            plus_vals.append(float(plus_ndcg))

        macro_full = float(np.mean(full_vals))
        macro_plus = float(np.mean(plus_vals))
        macro_diff = float(macro_plus - macro_full)
        per_repeat_macro_rows.append(
            {
                "repeat_index": repeat_idx,
                "macro_full_ndcg": macro_full,
                "macro_plus_ndcg": macro_plus,
                "macro_diff_plus_minus_full": macro_diff,
            }
        )
        print(
            f"  repeat={repeat_idx:>2} | macro_full={macro_full:.6f} | "
            f"macro_plus={macro_plus:.6f} | diff={macro_diff:+.6f}"
        )

    print("\n[4/4] Computing tests and writing outputs ...")
    out_dir = os.path.join(ablation_dir, f"statistical_full_vs_plus{args.plus_index}")
    ensure_dir(out_dir)

    repeat_dataset_csv = os.path.join(out_dir, "full_vs_plus11_per_repeat_per_dataset.csv")
    repeat_macro_csv = os.path.join(out_dir, "full_vs_plus11_per_repeat_macro.csv")
    tests_csv = os.path.join(out_dir, "full_vs_plus11_stat_tests.csv")

    save_repeat_dataset_csv(per_repeat_dataset_rows, repeat_dataset_csv)
    save_repeat_macro_csv(per_repeat_macro_rows, repeat_macro_csv)

    summary_rows = []
    for dataset_name in datasets:
        diffs = [
            r["diff_plus_minus_full"]
            for r in per_repeat_dataset_rows
            if r["dataset"] == dataset_name
        ]
        summary_rows.append(summarize_differences(f"dataset:{dataset_name}", diffs))

    macro_diffs = [r["macro_diff_plus_minus_full"] for r in per_repeat_macro_rows]
    summary_rows.append(summarize_differences("macro", macro_diffs))

    save_test_summary_csv(summary_rows, tests_csv)

    print("\nStatistical analysis completed.")
    print(f"Per-repeat dataset CSV : {repeat_dataset_csv}")
    print(f"Per-repeat macro CSV   : {repeat_macro_csv}")
    print(f"Tests summary CSV      : {tests_csv}")
    print("\nSummary:")
    for row in summary_rows:
        print(
            f"  {row['name']:<18} | mean_diff={row['mean_diff']:+.6f} | "
            f"t={row['t_stat']:+.4f} | p(two)={row['p_two_sided']:.6g} | "
            f"CI95=[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]"
        )

    if scipy_stats is None:
        print("\n[WARN] scipy is not available. p-values and confidence intervals may be NaN.")


if __name__ == "__main__":
    main()
