"""cleanup_old_artifacts.py

Conservative cleanup utility for thesis artifacts.

Design goals:
- Never delete expensive-to-build embedding/index artifacts.
- Keep files produced by the current implementation.
- Keep select legacy-equivalent outputs where naming changed but data meaning is the same.
- Dry-run by default; deletion requires --apply.

Usage:
  python src/cleanup_old_artifacts.py --config config.yaml
  python src/cleanup_old_artifacts.py --config config.yaml --apply
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from typing import Iterable, List, Set, Tuple

# Ensure project root is importable regardless of launch location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import get_config_path, load_config, model_short_name


def _match_any(path_posix: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path_posix, pat) for pat in patterns)


def _collect_files(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def _discover_model_namespaces(processed_root: Path, results_root: Path) -> List[str]:
    """Discover model namespace directories under processed/results roots."""
    names: Set[str] = set()

    if processed_root.exists() and processed_root.is_dir():
        for child in processed_root.iterdir():
            if child.is_dir():
                names.add(child.name)

    if results_root.exists() and results_root.is_dir():
        for child in results_root.iterdir():
            if child.is_dir():
                names.add(child.name)

    return sorted(names)


def _build_processed_keep_patterns(datasets: List[str]) -> Tuple[List[str], List[str]]:
    """Return (protected_expensive_patterns, keep_patterns) for processed/<model>."""
    protected: List[str] = []
    keep: List[str] = []

    # Legacy-equivalent cache locations (safe to preserve).
    keep.extend([
        "routing_cache/query_feature_label_cache.pkl",
        "routing_cache/query_feature_label_cache.csv",
        "query_feature_cache/query_feature_label_cache.pkl",
        "query_feature_cache/query_feature_label_cache.csv",
    ])

    for ds in datasets:
        # Core exported dataset files.
        keep.extend([
            f"{ds}/corpus.jsonl",
            f"{ds}/queries.jsonl",
            f"{ds}/qrels.tsv",
        ])

        # Expensive dense embeddings and ids.
        protected.extend([
            f"{ds}/corpus_embeddings.pt",
            f"{ds}/corpus_ids.pkl",
            f"{ds}/query_vectors.pt",
            f"{ds}/query_ids.pkl",
        ])

        # Expensive sparse/tokenization/index artifacts.
        protected.extend([
            f"{ds}/tokenized_corpus_stem_*.jsonl",
            f"{ds}/tokenized_queries_stem_*.jsonl",
            f"{ds}/query_tokens_stem_*.pkl",
            f"{ds}/word_freq_index_stem_*.pkl",
            f"{ds}/doc_freq_index_stem_*.pkl",
            f"{ds}/bm25_k1_*_b_*_stem_*.pkl",
            f"{ds}/bm25_k1_*_b_*_stem_*_doc_ids.pkl",
            f"{ds}/bm25_k1_*_b_*_stem_*_results.pkl",
            f"{ds}/bm25_k1_*_b_*_stem_*_topk_*_results.pkl",
            f"{ds}/dense_results_topk_*.pkl",
        ])

    # Protected files are also implicitly kept.
    keep.extend(protected)
    return protected, keep


def _build_results_keep_patterns() -> List[str]:
    """Return keep patterns for results root and results/<model>."""
    keep: List[str] = [
        # Global BM25 optimization outputs (results root).
        "bm25_optimization.csv",
        "bm25_optimization_macro.csv",
        "bm25_optimization_best.json",

        # LOODO benchmark outputs.
        "supervised_routing/loodo_per_repeat_results.csv",
        "supervised_routing/per_dataset_results.csv",
        "supervised_routing/per_dataset_summary.csv",
        "supervised_routing/loodo_macro_summary.csv",
        # Legacy-equivalent name for older runs.
        "supervised_routing/macro_summary.csv",
        "supervised_routing/predicted_alphas.csv",
        "supervised_routing/fold_model_effects.csv",
        "supervised_routing/fold_normalization_stats.csv",
        "supervised_routing/router_model_metadata.json",
        "supervised_routing/fold_models/fold_*.pkl",
        "supervised_routing/plots/per_dataset_comparison.png",
        "supervised_routing/plots/macro_average_comparison.png",
        "supervised_routing/plots/alpha_distribution.png",

        # Within-dataset benchmark outputs.
        "within_dataset_routing/per_repeat_results.csv",
        "within_dataset_routing/per_dataset_summary.csv",
        "within_dataset_routing/macro_summary.csv",
        "within_dataset_routing/predicted_alphas.csv",
        "within_dataset_routing/model_effects.csv",
        "within_dataset_routing/normalization_stats.csv",
        "within_dataset_routing/alpha_summary.csv",
        "within_dataset_routing/router_model_metadata.json",
        "within_dataset_routing/plots/within_dataset_comparison.png",
        "within_dataset_routing/plots/within_dataset_alpha_distribution.png",

        # XGBoost optimization outputs.
        "xgboost_optimization/within_dataset/trial_results.csv",
        "xgboost_optimization/within_dataset/summary.csv",
        "xgboost_optimization/within_dataset/best_params_per_dataset.json",
        "xgboost_optimization/within_dataset/best_params_per_dataset.yaml",
        "xgboost_optimization/within_dataset/search_metadata.json",
        "xgboost_optimization/loodo/trial_results.csv",
        "xgboost_optimization/loodo/summary.csv",
        "xgboost_optimization/loodo/best_params_per_dataset.json",
        "xgboost_optimization/loodo/best_params_per_dataset.yaml",
        "xgboost_optimization/loodo/search_metadata.json",

        # Ablation outputs.
        "ablation/lofo_per_dataset_delta.csv",
        "ablation/lofo_macro_delta.csv",
        "ablation/lofo_macro_delta_plot.png",

        # Feature-ladder comparison outputs.
        "ablation/negative_feature_subset_comparison/kept_negative_delta_features.csv",
        "ablation/negative_feature_subset_comparison/model_ladder_plan.csv",
        "ablation/negative_feature_subset_comparison/model_ladder_per_dataset.csv",
        "ablation/negative_feature_subset_comparison/model_ladder_macro.csv",
        "ablation/negative_feature_subset_comparison/model_ladder_macro_delta_vs_full.png",
        "ablation/negative_feature_subset_comparison/model_ladder_dataset_delta_vs_full.png",

        # Statistical analysis outputs.
        "ablation/statistical_full_vs_plus*/full_vs_plus*_per_repeat_per_dataset.csv",
        "ablation/statistical_full_vs_plus*/full_vs_plus*_per_repeat_macro.csv",
        "ablation/statistical_full_vs_plus*/full_vs_plus*_stat_tests.csv",

        # XGBoost analysis outputs (current).
        "analysis/shap_values.csv",
        "analysis/alpha_analysis.csv",
        "analysis/feature_alpha_correlation.csv",
        "analysis/feature_label_correlation.csv",
        "analysis/shap_feature_ranking.png",
        "analysis/all_features_vs_soft_label.png",
        "analysis/alpha_distribution.png",
        "analysis/alpha_per_dataset.png",
        "analysis/alpha_vs_feature_*.png",

        # Legacy-equivalent output where name changed but semantics match.
        "analysis/shap_bar.png",
    ]

    return keep


def _classify_files(
    files: List[Path],
    root: Path,
    keep_patterns: List[str],
    protected_patterns: List[str] | None = None,
) -> Tuple[List[Path], List[Path], List[Path]]:
    protected_patterns = protected_patterns or []

    kept: List[Path] = []
    protected: List[Path] = []
    to_delete: List[Path] = []

    for f in files:
        rel = f.relative_to(root).as_posix()

        if _match_any(rel, protected_patterns):
            protected.append(f)
            kept.append(f)
            continue

        if _match_any(rel, keep_patterns):
            kept.append(f)
        else:
            to_delete.append(f)

    return kept, protected, to_delete


def _remove_files(files: List[Path]) -> int:
    removed = 0
    for f in files:
        try:
            f.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def _remove_empty_dirs(root: Path) -> int:
    """Remove empty directories under root, deepest first."""
    if not root.exists() or not root.is_dir():
        return 0

    removed = 0
    dirs = sorted([d for d in root.rglob("*") if d.is_dir()], key=lambda p: len(p.parts), reverse=True)
    for d in dirs:
        try:
            d.rmdir()
            removed += 1
        except OSError:
            # Not empty or cannot remove.
            continue
    return removed


def _print_list(title: str, files: List[Path], root: Path, limit: int) -> None:
    print(title)
    if not files:
        print("  (none)")
        return
    for p in files[:limit]:
        print(f"  - {p.relative_to(root).as_posix()}")
    if len(files) > limit:
        print(f"  ... and {len(files) - limit} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively clean old artifacts while preserving expensive indexes/embeddings "
            "and outputs used by current implementation. Dry-run by default."
        )
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, only a dry-run report is produced.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=200,
        help="Maximum number of candidate-deletion paths to print.",
    )
    parser.add_argument(
        "--prune-empty-dirs",
        action="store_true",
        help="After deletion, remove empty directories under cleaned roots.",
    )
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help="Include processed_data/<current_model> in cleanup scan.",
    )
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Include results root and results/<current_model> in cleanup scan.",
    )
    parser.add_argument(
        "--model-short-name",
        default=None,
        help=(
            "Optional model namespace override (e.g., all-MiniLM-L6-v2). "
            "Default uses embeddings.model_name from config."
        ),
    )
    parser.add_argument(
        "--all-model-namespaces",
        action="store_true",
        help="Scan all discovered model namespaces under processed/results roots.",
    )
    args = parser.parse_args()

    # If neither scope flag is provided, scan both by default.
    include_processed = args.include_processed or (not args.include_processed and not args.include_results)
    include_results = args.include_results or (not args.include_processed and not args.include_results)

    u.CONFIG_PATH = args.config
    cfg = load_config()

    datasets = [str(d) for d in cfg.get("datasets", [])]
    if not datasets:
        raise ValueError("No datasets configured. Refusing cleanup because scope cannot be validated safely.")

    configured_short_model = model_short_name(cfg["embeddings"]["model_name"])
    processed_root = Path(get_config_path(cfg, "processed_folder", "data/processed_data")).resolve()
    results_root = Path(get_config_path(cfg, "results_folder", "data/results")).resolve()

    discovered_models = _discover_model_namespaces(processed_root, results_root)
    if args.all_model_namespaces:
        target_models = discovered_models if discovered_models else [configured_short_model]
    else:
        target_models = [args.model_short_name or configured_short_model]

    total_candidates: List[Path] = []

    print("=" * 72)
    print("Cleanup plan (conservative)")
    print(f"Configured model namespace: {configured_short_model}")
    print(f"Target model namespaces: {', '.join(target_models)}")
    if discovered_models:
        print(f"Discovered model namespaces: {', '.join(discovered_models)}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print("=" * 72)

    if include_processed:
        protected_patterns, processed_keep = _build_processed_keep_patterns(datasets)
        for model_name in target_models:
            processed_model_dir = processed_root / model_name
            files = _collect_files(processed_model_dir)
            kept, protected, to_delete = _classify_files(
                files,
                processed_model_dir,
                keep_patterns=processed_keep,
                protected_patterns=protected_patterns,
            )
            total_candidates.extend(to_delete)

            print(f"\n[processed_data] model={model_name} root: {processed_model_dir}")
            print(f"  files scanned      : {len(files)}")
            print(f"  protected kept     : {len(protected)}")
            print(f"  kept by allowlist  : {len(kept) - len(protected)}")
            print(f"  delete candidates  : {len(to_delete)}")
            _print_list("  candidate examples:", to_delete, processed_model_dir, args.max_print)

    if include_results:
        results_keep = _build_results_keep_patterns()

        # Model-scoped results.
        for model_name in target_models:
            results_model_dir = results_root / model_name
            model_files = _collect_files(results_model_dir)
            kept_m, _, to_delete_m = _classify_files(
                model_files,
                results_model_dir,
                keep_patterns=results_keep,
                protected_patterns=[],
            )
            total_candidates.extend(to_delete_m)

            print(f"\n[results/model] model={model_name} root: {results_model_dir}")
            print(f"  files scanned      : {len(model_files)}")
            print(f"  kept by allowlist  : {len(kept_m)}")
            print(f"  delete candidates  : {len(to_delete_m)}")
            _print_list("  candidate examples:", to_delete_m, results_model_dir, args.max_print)

        # Global results root files (only top-level files in keep list are preserved).
        global_files = [p for p in _collect_files(results_root) if p.parent == results_root]
        global_keep = [pat for pat in results_keep if "/" not in pat]
        kept_g, _, to_delete_g = _classify_files(
            global_files,
            results_root,
            keep_patterns=global_keep,
            protected_patterns=[],
        )
        total_candidates.extend(to_delete_g)

        print(f"\n[results/global] root: {results_root}")
        print(f"  files scanned      : {len(global_files)}")
        print(f"  kept by allowlist  : {len(kept_g)}")
        print(f"  delete candidates  : {len(to_delete_g)}")
        _print_list("  candidate examples:", to_delete_g, results_root, args.max_print)

    # Deduplicate and sort for deterministic behavior.
    unique_candidates = sorted(set(total_candidates), key=lambda p: str(p).lower())

    print("\n" + "=" * 72)
    print(f"Total delete candidates: {len(unique_candidates)}")

    if not args.apply:
        print("Dry-run only. No files were deleted.")
        print("Re-run with --apply to execute deletions.")
        print("=" * 72)
        return

    removed_files = _remove_files(unique_candidates)
    print(f"Deleted files: {removed_files}")

    removed_dirs = 0
    if args.prune_empty_dirs:
        if include_processed:
            for model_name in target_models:
                removed_dirs += _remove_empty_dirs(processed_root / model_name)
        if include_results:
            for model_name in target_models:
                removed_dirs += _remove_empty_dirs(results_root / model_name)
            removed_dirs += _remove_empty_dirs(results_root)
        print(f"Removed empty directories: {removed_dirs}")

    print("Cleanup completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
