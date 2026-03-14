"""
retrieval.py -- Run only retrieval + evaluation phases for cached datasets.

Usage:
    python src/retrieval.py              # run with default config.yaml
    python src/retrieval.py --config my_config.yaml

This script expects preprocessing/indexing/embedding artifacts to already exist
under data/results/<model_short>/<dataset>/ (or /merged/ in merge mode).
It executes:
  - BM25 retrieval (step 6)
  - Dense retrieval (step 8)
  - Evaluation/fusion (step 9)
  - Save results CSV/chart + retrieval timing CSV
"""

import argparse
import os
import sys
import time
import torch

# Ensure project root is on sys.path and cwd is project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.utils import (
    load_config,
    load_pickle,
    save_pickle,
    file_exists,
    load_queries,
    load_qrels,
    model_short_name,
    save_results_csv,
    save_results_chart,
    save_timing_csv,
)
from src.pipeline import (
    run_bm25_retrieval,
    run_dense_retrieval,
    evaluate_all_methods,
)


def _missing_paths(paths):
    """Return the subset of *paths* that do not exist."""
    return [p for p in paths if not file_exists(p)]


def run_retrieval_eval_for_dir(ds_dir, label, cfg, device):
    """Run retrieval + evaluation using cached artifacts in *ds_dir*."""
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    top_k = cfg["benchmark"]["top_k"]
    dense_cfg = cfg["dense_search"]
    short_model = model_short_name(cfg["embeddings"]["model_name"])

    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
    qrels_tsv = os.path.join(ds_dir, "qrels.tsv")

    bm25_pkl = os.path.join(ds_dir, "bm25_index.pkl")
    bm25_docids_pkl = os.path.join(ds_dir, "bm25_doc_ids.pkl")
    bm25_results_pkl = os.path.join(ds_dir, "bm25_results.pkl")

    corpus_emb_pt = os.path.join(ds_dir, "corpus_embeddings.pt")
    corpus_ids_pkl = os.path.join(ds_dir, "corpus_ids.pkl")
    query_vectors_pt = os.path.join(ds_dir, "query_vectors.pt")
    query_ids_pkl = os.path.join(ds_dir, "query_ids.pkl")
    dense_results_pkl = os.path.join(ds_dir, "dense_results.pkl")

    word_freq_pkl = os.path.join(ds_dir, "word_freq_index.pkl")

    results_csv = os.path.join(ds_dir, "results.csv")
    results_png = os.path.join(ds_dir, "results.png")
    timing_csv = os.path.join(ds_dir, "timing_retrieval.csv")

    print(f"\n{'=' * 60}")
    print(f"  Retrieval-only: {label}")
    print(f"  Output        : {ds_dir}")
    print(f"{'=' * 60}")

    # Mandatory shared inputs
    missing_base = _missing_paths([queries_jsonl, qrels_tsv, word_freq_pkl])
    if missing_base:
        print("  [ERROR] Missing required cached artifacts:")
        for p in missing_base:
            print(f"    - {p}")
        print("  Run src/pipeline.py first to generate these artifacts.")
        return

    queries = load_queries(queries_jsonl)
    qrels = load_qrels(qrels_tsv)
    global_counts, total_corpus_tokens = load_pickle(word_freq_pkl)

    timings = []

    # Step 6: BM25 retrieval
    t0 = time.time()
    if file_exists(bm25_results_pkl):
        print("[Step 6] BM25 results exist. Loading ...")
        bm25_results = load_pickle(bm25_results_pkl)
    else:
        print("[Step 6] Running BM25 retrieval ...")
        missing_bm25 = _missing_paths([bm25_pkl, bm25_docids_pkl])
        if missing_bm25:
            print("  [ERROR] Cannot run BM25 retrieval, missing:")
            for p in missing_bm25:
                print(f"    - {p}")
            return
        bm25 = load_pickle(bm25_pkl)
        bm25_doc_ids = load_pickle(bm25_docids_pkl)
        bm25_results = run_bm25_retrieval(
            bm25, bm25_doc_ids, queries, stemmer_lang, top_k
        )
        save_pickle(bm25_results, bm25_results_pkl)
    timings.append(("Step 6: BM25 retrieval", time.time() - t0))

    # Step 8: Dense retrieval
    t0 = time.time()
    if file_exists(dense_results_pkl):
        print("[Step 8] Dense results exist. Loading ...")
        dense_results = load_pickle(dense_results_pkl)
    else:
        print("[Step 8] Running dense retrieval ...")
        missing_dense = _missing_paths(
            [corpus_emb_pt, corpus_ids_pkl, query_vectors_pt, query_ids_pkl]
        )
        if missing_dense:
            print("  [ERROR] Cannot run dense retrieval, missing:")
            for p in missing_dense:
                print(f"    - {p}")
            return
        corpus_embeddings = torch.load(corpus_emb_pt, weights_only=True)
        corpus_ids = load_pickle(corpus_ids_pkl)
        query_vectors = torch.load(query_vectors_pt, weights_only=True)
        query_ids = load_pickle(query_ids_pkl)
        dense_results = run_dense_retrieval(
            query_vectors,
            query_ids,
            corpus_embeddings,
            corpus_ids,
            top_k,
            dense_cfg["corpus_chunk_size"],
            device=device,
            query_chunk_size=dense_cfg.get("query_chunk_size", 100),
        )
        save_pickle(dense_results, dense_results_pkl)
    timings.append(("Step 8: Dense retrieval", time.time() - t0))

    # Step 9: Evaluation
    print("[Step 9] Evaluating fusion methods ...")
    t0 = time.time()
    method_scores = evaluate_all_methods(
        bm25_results,
        dense_results,
        qrels,
        cfg["benchmark"],
        queries,
        stemmer_lang,
        global_counts,
        total_corpus_tokens,
    )
    timings.append(("Step 9: Evaluation (all fusion methods)", time.time() - t0))

    print(f"\n  {'Method':<32s} {'NDCG@10':>8s}")
    print(f"  {'-'*32} {'-'*8}")
    for method, ndcg in method_scores:
        print(f"  {method:<32s} {ndcg:>8.4f}")

    # Step 10: Save outputs
    print("\n[Step 10] Saving results ...")
    t0 = time.time()
    save_results_csv(method_scores, results_csv)
    print(f"  Results saved to {results_csv}")
    save_results_chart(method_scores, results_png, label, short_model)
    timings.append(("Step 10: Save results", time.time() - t0))

    save_timing_csv(timings, timing_csv)
    print(f"  Retrieval timing saved to {timing_csv}")

    print(f"\n  {'Step':<45s} {'Time (s)':>10s}")
    print(f"  {'-'*45} {'-'*10}")
    for label_t, secs in timings:
        print(f"  {label_t:<45s} {secs:>10.2f}")
    total_time = sum(t for _, t in timings)
    print(f"  {'-'*45} {'-'*10}")
    print(f"  {'Total':<45s} {total_time:>10.2f}")

    print(f"\n  Finished retrieval-only for {label}.")


def main():
    parser = argparse.ArgumentParser(
        description="RAG-LLM Retrieval + Evaluation (cached artifacts only)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    # Override config path consumed by src.utils.load_config
    import src.utils as _u
    _u.CONFIG_PATH = args.config

    cfg = load_config()
    datasets = cfg.get("datasets", [])
    if not datasets:
        print("No datasets specified in config.yaml. Nothing to do.")
        sys.exit(0)

    merge_mode = cfg.get("merge", False)
    model_name = cfg["embeddings"]["model_name"]
    short_model = model_short_name(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Model  : {model_name}")
    print(f"Mode   : {'MERGE' if merge_mode else 'PER-DATASET'}")

    if merge_mode:
        merged_dir = os.path.join(cfg["paths"]["results_folder"], short_model, "merged")
        run_retrieval_eval_for_dir(merged_dir, "merged", cfg, device)
    else:
        print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
        for ds_name in datasets:
            ds_dir = os.path.join(cfg["paths"]["results_folder"], short_model, ds_name)
            run_retrieval_eval_for_dir(ds_dir, ds_name, cfg, device)

    print("\n" + "=" * 60)
    print("  Retrieval-only run complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
