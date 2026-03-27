"""
preproces.py

Builds preprocessing artifacts for configured datasets and caches them on disk.
The script always skips steps that are already available in processed_data.

Artifacts built here:
  - corpus.jsonl / queries.jsonl / qrels.tsv
  - tokenized_corpus.jsonl
  - tokenized_queries.jsonl
  - bm25_index.pkl + bm25_doc_ids.pkl
    - word_freq_index.pkl
    - doc_freq_index.pkl
  - corpus_embeddings.pt + corpus_ids.pkl
  - query_vectors.pt + query_ids.pkl
  - query_tokens.pkl
"""

import argparse
import math
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import torch
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Make sure project root is importable regardless of launch location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import (
    load_config,
    ensure_dir,
    file_exists,
    load_pickle,
    save_pickle,
    write_corpus_jsonl,
    write_queries_jsonl,
    write_qrels_tsv,
    load_queries,
    download_beir_dataset,
    load_beir_dataset,
    model_short_name,
    stem_and_tokenize,
    count_lines,
    load_corpus_batch_generator,
    stem_batch_worker,
    _init_worker,
)


# Use most CPU cores for local preprocessing and encoding support routines.
_N_CORES = os.cpu_count() or 4
torch.set_num_threads(_N_CORES)
torch.set_num_interop_threads(1)

# Module-level flag used by the encoding OOM fallback helper.
_gpu_failed = False


def _is_nonempty_file(path):
    """Return True when path exists and has non-zero size."""
    return file_exists(path) and os.path.getsize(path) > 0


def preprocess_corpus(
    corpus_jsonl,
    output_jsonl,
    stemmer_lang,
    use_stemming,
    batch_size=512,
):
    """Stem-tokenize the corpus JSONL using multiprocessing and cache to disk."""
    ensure_dir(os.path.dirname(output_jsonl))
    n_workers = max(1, _N_CORES - 1)
    max_pending = n_workers * 3
    total_batches = math.ceil(count_lines(corpus_jsonl) / batch_size)

    with open(output_jsonl, "w", encoding="utf-8") as out:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(stemmer_lang, use_stemming),
        ) as pool:
            pending = {}
            completed = {}
            batch_idx = 0
            next_to_write = 0
            pbar = tqdm(
                total=total_batches,
                desc="  Preprocessing corpus",
                dynamic_ncols=True,
            )

            for batch_ids, batch_texts in load_corpus_batch_generator(corpus_jsonl, batch_size):
                fut = pool.submit(stem_batch_worker, (batch_ids, batch_texts))
                pending[fut] = batch_idx
                batch_idx += 1

                while len(pending) >= max_pending:
                    done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                    for fut_done in done:
                        idx = pending.pop(fut_done)
                        completed[idx] = fut_done.result()

                    while next_to_write in completed:
                        for line in completed.pop(next_to_write):
                            out.write(line + "\n")
                        pbar.update(1)
                        next_to_write += 1

            while pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for fut_done in done:
                    idx = pending.pop(fut_done)
                    completed[idx] = fut_done.result()

                while next_to_write in completed:
                    for line in completed.pop(next_to_write):
                        out.write(line + "\n")
                    pbar.update(1)
                    next_to_write += 1

            pbar.close()


def build_bm25_and_word_freq_index(tokenized_corpus_jsonl, k1, b):
    """Build BM25 index plus corpus-wide token frequency index in one pass."""
    from rank_bm25 import BM25Okapi

    doc_ids = []
    tokenized_docs = []
    global_counts = {}
    total_tokens = 0

    num_lines = count_lines(tokenized_corpus_jsonl)
    with open(tokenized_corpus_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="  Loading tokenized corpus", dynamic_ncols=True):
            d = json.loads(line)
            doc_ids.append(d["_id"])
            tokens = d["tokens"]
            tokenized_docs.append(tokens)
            for t in tokens:
                global_counts[t] = global_counts.get(t, 0) + 1
            total_tokens += len(tokens)

    print(f"  Building BM25 index over {len(doc_ids):,} documents ...")
    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    print(
        f"  Vocabulary size: {len(global_counts):,} unique tokens, "
        f"{total_tokens:,} total occurrences."
    )
    return bm25, doc_ids, global_counts, total_tokens


def build_doc_freq_index(tokenized_corpus_jsonl):
    """Build document frequency index: token -> number of docs containing token."""
    doc_freq = {}
    total_docs = 0

    num_lines = count_lines(tokenized_corpus_jsonl)
    with open(tokenized_corpus_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="  Building doc frequency", dynamic_ncols=True):
            d = json.loads(line)
            total_docs += 1
            for token in set(d.get("tokens", [])):
                doc_freq[token] = doc_freq.get(token, 0) + 1

    print(f"  Document-frequency index: {len(doc_freq):,} unique tokens over {total_docs:,} docs.")
    return doc_freq, total_docs


def _encode_with_oom_retry(model, texts, device, batch_size):
    """Encode texts and fallback to smaller batches or CPU on CUDA errors."""
    global _gpu_failed

    results = []
    start = 0
    current_bs = batch_size
    current_device = "cpu" if _gpu_failed else device

    while start < len(texts):
        end = min(start + current_bs, len(texts))
        sub_batch = texts[start:end]
        try:
            embs = model.encode(
                sub_batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=current_device,
            )
            results.append(embs.cpu())
            start = end
            if current_device == device:
                current_bs = batch_size
        except torch.cuda.OutOfMemoryError:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            current_bs = max(1, current_bs // 2)
            print(f"\n  [OOM] Reduced sub-batch size to {current_bs}")
            if current_bs == 1 and end - start == 1:
                raise
        except Exception as exc:
            _AcceleratorError = getattr(torch, "AcceleratorError", None)
            is_cuda_error = (
                (_AcceleratorError is not None and isinstance(exc, _AcceleratorError))
                or ("cuda" in str(exc).lower())
            )
            if is_cuda_error and current_device != "cpu":
                print("\n  [CUDA ERROR] Unrecoverable GPU error -- falling back to CPU.")
                print(f"  ({type(exc).__name__}: {exc})")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                _gpu_failed = True
                current_device = "cpu"
                current_bs = batch_size
            else:
                raise

    return results


def build_corpus_embeddings(corpus_jsonl, model, batch_size, device):
    """Encode all corpus documents and return embeddings tensor and doc ids."""
    all_ids, all_embs = [], []
    total = count_lines(corpus_jsonl)

    if _gpu_failed or device == "cpu":
        print("  Encoding device : cpu")
    elif torch.cuda.is_available():
        used_gb = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Encoding device : cuda  |  VRAM {used_gb:.1f} / {total_gb:.1f} GB allocated")

    for batch_ids, batch_texts in tqdm(
        load_corpus_batch_generator(corpus_jsonl, batch_size),
        total=math.ceil(total / batch_size),
        desc="  Encoding corpus",
        dynamic_ncols=True,
    ):
        embs_list = _encode_with_oom_retry(model, batch_texts, device, batch_size)
        all_ids.extend(batch_ids)
        all_embs.extend(embs_list)

    if not all_embs:
        raise ValueError(
            "No corpus embeddings were produced. The corpus appears empty or malformed."
        )

    return torch.cat(all_embs, dim=0), all_ids


def build_dense_query_vectors(queries, model, batch_size, device):
    """Encode queries and return embeddings tensor and query ids."""
    qids = list(queries.keys())
    qtexts = [queries[q] for q in qids]
    all_embs = []

    for i in tqdm(range(0, len(qtexts), batch_size), desc="  Encoding queries", dynamic_ncols=True):
        batch = qtexts[i: i + batch_size]
        embs_list = _encode_with_oom_retry(model, batch, device, batch_size)
        all_embs.extend(embs_list)

    if not all_embs:
        raise ValueError(
            "No query embeddings were produced. The query set appears empty or malformed."
        )

    return torch.cat(all_embs, dim=0), qids


def preprocess_queries(
    queries_jsonl,
    tokenized_queries_jsonl,
    query_tokens_pkl,
    stemmer_lang,
    use_stemming,
):
    """Tokenize/stem queries and cache both JSONL and dict-by-id forms."""
    if file_exists(tokenized_queries_jsonl) and file_exists(query_tokens_pkl):
        cache_ok = False
        try:
            token_map = load_pickle(query_tokens_pkl)
            cache_ok = isinstance(token_map, dict) and _is_nonempty_file(tokenized_queries_jsonl)
        except Exception as exc:
            print(
                "  [WARN] Query preprocessing cache unreadable; rebuilding. "
                f"({type(exc).__name__}: {exc})"
            )
        if cache_ok:
            print("  Query preprocessing cache exists. Skipping.")
            return
        print("  [WARN] Query preprocessing cache is incomplete/corrupted; rebuilding.")

    queries = load_queries(queries_jsonl)
    stemmer = SnowballStemmer(stemmer_lang) if use_stemming else None
    token_map = {}

    with open(tokenized_queries_jsonl, "w", encoding="utf-8") as out:
        for qid, qtext in queries.items():
            tokens = stem_and_tokenize(qtext, stemmer)
            token_map[qid] = tokens
            out.write(json.dumps({"_id": qid, "tokens": tokens}) + "\n")

    save_pickle(token_map, query_tokens_pkl)


def run_for_dataset(dataset_name, cfg, model, device):
    global _gpu_failed
    _gpu_failed = False

    datasets_folder = u.get_config_path(cfg, "datasets_folder", "data/datasets")
    processed_folder = u.get_config_path(cfg, "processed_folder", "data/processed_data")

    short_model = model_short_name(cfg["embeddings"]["model_name"])
    ds_dir = os.path.join(processed_folder, short_model, dataset_name)
    ensure_dir(ds_dir)

    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    emb_batch_size = cfg["embeddings"]["batch_size"]
    bm25_params = u.get_bm25_params(cfg)
    sparse_paths = u.bm25_artifact_paths(
        ds_dir,
        bm25_params["k1"],
        bm25_params["b"],
        bm25_params["use_stemming"],
    )

    corpus_jsonl = os.path.join(ds_dir, "corpus.jsonl")
    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
    qrels_tsv = os.path.join(ds_dir, "qrels.tsv")
    tokenized_corpus_jsonl = sparse_paths["tokenized_corpus_jsonl"]
    tokenized_queries_jsonl = sparse_paths["tokenized_queries_jsonl"]
    query_tokens_pkl = sparse_paths["query_tokens_pkl"]
    bm25_pkl = sparse_paths["bm25_pkl"]
    bm25_docids_pkl = sparse_paths["bm25_docids_pkl"]
    word_freq_pkl = sparse_paths["word_freq_pkl"]
    doc_freq_pkl = sparse_paths["doc_freq_pkl"]
    corpus_emb_pt = os.path.join(ds_dir, "corpus_embeddings.pt")
    corpus_ids_pkl = os.path.join(ds_dir, "corpus_ids.pkl")
    query_vectors_pt = os.path.join(ds_dir, "query_vectors.pt")
    query_ids_pkl = os.path.join(ds_dir, "query_ids.pkl")

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"Processed output: {ds_dir}")
    print(
        "BM25 config: "
        f"k1={bm25_params['k1']}, b={bm25_params['b']}, "
        f"use_stemming={bm25_params['use_stemming']}"
    )
    print(f"BM25 signature: {sparse_paths['bm25_signature']}")
    print(f"{'=' * 60}")

    # 1) Ensure dataset is available locally
    dataset_path = download_beir_dataset(dataset_name, datasets_folder)
    if dataset_path is None:
        print("  [ERROR] Dataset download/verification failed. Skipping dataset.")
        return

    # 2) Export corpus/queries/qrels once
    if file_exists(corpus_jsonl) and file_exists(queries_jsonl) and file_exists(qrels_tsv):
        print("[1/6] Corpus/queries/qrels already cached. Skipping.")
    else:
        print("[1/6] Loading BEIR data and writing corpus/queries/qrels ...")
        corpus, queries, qrels, split = load_beir_dataset(dataset_path)
        if corpus is None:
            print("  [ERROR] Failed to load BEIR dataset. Skipping dataset.")
            return
        print(f"  Split: {split} | Corpus: {len(corpus):,} | Queries: {len(queries):,}")
        write_corpus_jsonl(corpus, corpus_jsonl)
        write_queries_jsonl(queries, queries_jsonl)
        write_qrels_tsv(qrels, qrels_tsv)

    # 3) Preprocess corpus and queries
    if _is_nonempty_file(tokenized_corpus_jsonl):
        print("[2/6] Tokenized corpus exists. Skipping.")
    else:
        if file_exists(tokenized_corpus_jsonl):
            print("[2/6] Tokenized corpus cache is empty/incomplete; rebuilding.")
        print("[2/6] Tokenizing/stemming corpus ...")
        preprocess_corpus(
            corpus_jsonl,
            tokenized_corpus_jsonl,
            stemmer_lang,
            use_stemming=bm25_params["use_stemming"],
        )

    print("[3/6] Preprocessing queries ...")
    preprocess_queries(
        queries_jsonl,
        tokenized_queries_jsonl,
        query_tokens_pkl,
        stemmer_lang,
        bm25_params["use_stemming"],
    )

    # 4) BM25 + frequency indexes
    has_index = (
        _is_nonempty_file(bm25_pkl)
        and _is_nonempty_file(bm25_docids_pkl)
        and _is_nonempty_file(word_freq_pkl)
        and _is_nonempty_file(doc_freq_pkl)
    )
    if has_index:
        print("[4/6] BM25 + frequency indexes exist. Skipping.")
    else:
        print("[4/6] Building BM25 + frequency indexes ...")
        bm25, bm25_doc_ids, global_counts, total_corpus_tokens = build_bm25_and_word_freq_index(
            tokenized_corpus_jsonl,
            k1=bm25_params["k1"],
            b=bm25_params["b"],
        )
        doc_freq, total_docs = build_doc_freq_index(tokenized_corpus_jsonl)
        save_pickle(bm25, bm25_pkl)
        save_pickle(bm25_doc_ids, bm25_docids_pkl)
        save_pickle((global_counts, total_corpus_tokens), word_freq_pkl)
        save_pickle((doc_freq, total_docs), doc_freq_pkl)

    # 5) Corpus embeddings
    if _is_nonempty_file(corpus_emb_pt) and _is_nonempty_file(corpus_ids_pkl):
        print("[5/6] Corpus embeddings exist. Skipping.")
    else:
        if file_exists(corpus_emb_pt) or file_exists(corpus_ids_pkl):
            print("[5/6] Corpus embedding cache is incomplete; rebuilding.")
        print("[5/6] Encoding corpus ...")
        corpus_embeddings, corpus_ids = build_corpus_embeddings(
            corpus_jsonl,
            model,
            emb_batch_size,
            device,
        )
        torch.save(corpus_embeddings, corpus_emb_pt)
        save_pickle(corpus_ids, corpus_ids_pkl)

    # 6) Query embeddings
    if _is_nonempty_file(query_vectors_pt) and _is_nonempty_file(query_ids_pkl):
        print("[6/6] Query embeddings exist. Skipping.")
    else:
        if file_exists(query_vectors_pt) or file_exists(query_ids_pkl):
            print("[6/6] Query embedding cache is incomplete; rebuilding.")
        print("[6/6] Encoding queries ...")
        queries = load_queries(queries_jsonl)
        query_vectors, query_ids = build_dense_query_vectors(
            queries,
            model,
            emb_batch_size,
            device,
        )
        torch.save(query_vectors, query_vectors_pt)
        save_pickle(query_ids, query_ids_pkl)


def main():
    parser = argparse.ArgumentParser(
        description="Build cached preprocessing artifacts for configured datasets."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    datasets = cfg.get("datasets", [])
    if not datasets:
        print("No datasets configured. Nothing to preprocess.")
        return

    # Always ensure the three top-level data folders exist.
    datasets_folder = u.get_config_path(cfg, "datasets_folder", "data/datasets")
    results_folder = u.get_config_path(cfg, "results_folder", "data/results")
    processed_folder = u.get_config_path(cfg, "processed_folder", "data/processed_data")
    ensure_dir(datasets_folder)
    ensure_dir(results_folder)
    ensure_dir(processed_folder)

    model_name = cfg["embeddings"]["model_name"]
    routing_cfg = cfg.get("supervised_routing", {})
    use_cuda_if_available = bool(routing_cfg.get("use_cuda_if_available", True))
    device = "cuda" if (use_cuda_if_available and torch.cuda.is_available()) else "cpu"

    if torch.cuda.is_available() and not use_cuda_if_available:
        print("CUDA is available but disabled by supervised_routing.use_cuda_if_available=false.")

    print(f"Device : {device}")
    print(f"Model  : {model_name}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")

    print("\nLoading embedding model ...")
    model = SentenceTransformer(model_name, device=device)

    max_seq = cfg["embeddings"].get("max_seq_length")
    if max_seq is not None:
        model.max_seq_length = int(max_seq)

    for ds_name in datasets:
        run_for_dataset(ds_name, cfg, model, device)

    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()
