"""
pipeline.py -- Single entry point for the RAG-LLM hybrid retrieval benchmark.

Usage:
    python src/pipeline.py              # run with default config.yaml
    python src/pipeline.py --config my_config.yaml

The pipeline performs the following steps for every dataset
listed in config.yaml:
  1. Download / verify the BEIR dataset
  2. Load dataset and write local copies of corpus / queries / qrels
  3. Preprocess (stem & tokenize) the corpus for BM25
  4. Build the BM25 index and word-frequency index (single pass)
  5. Encode the corpus with a sentence-transformer model
  6. Run BM25 retrieval for all queries
  7. Encode queries with the same sentence-transformer
  8. Run dense retrieval for all queries
  9. Evaluate all fusion methods (BM25-only, Dense-only, RRF, 6×WRRF)
  10. Save CSV results and bar chart to the output folder

Intermediate artifacts (pickles, embeddings) are cached in
  data/results/<model_short_name>/<dataset_name>/
so re-running the script skips completed steps automatically.

A separate --merge mode concatenates multiple BEIR datasets into a
single combined corpus and runs the same pipeline on the merged data.
"""

import argparse
import json
import math
import os
import sys
import time
import logging
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util as st_util
from tqdm import tqdm

# ── Silence noisy HF / tokenizer download bars that overlap own tqdm bars
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# ── Maximise CPU parallelism for PyTorch / NumPy ──
_N_CORES = os.cpu_count() or 4
torch.set_num_threads(_N_CORES)
torch.set_num_interop_threads(1)   # sequential pipeline — inter-op unused

# Make sure the project root is on sys.path so that
# "from utils import ..." works regardless of CWD.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.utils import (
    load_config,
    ensure_dir,
    count_lines,
    save_pickle,
    load_pickle,
    file_exists,
    load_queries,
    load_qrels,
    load_corpus_batch_generator,
    download_beir_dataset,
    load_beir_dataset,
    initialize_output_files,
    append_corpus_to_jsonl,
    append_queries_to_jsonl,
    append_qrels_to_tsv,
    write_corpus_jsonl,
    write_queries_jsonl,
    write_qrels_tsv,
    model_short_name,
    stem_and_tokenize,
    stem_batch_worker,
    _init_worker,
    save_results_csv,
    save_results_chart,
    save_timing_csv,
)


# ============================================================
# 0. Shared math / scoring functions used by the fusion step
# ============================================================

def sigmoid(x, center, slope):
    """Standard sigmoid shifted by *center* with adjustable *slope*."""
    return 1.0 / (1.0 + math.exp(-slope * (x - center)))


def compute_query_metrics(query_tokens, global_counts, total_corpus_tokens,
                         laplace_alpha):
    """Compute CE, KLD and JSD between a query's word distribution and
    the global corpus word distribution.

    Parameters
    ----------
    query_tokens : list[str]
        Pre-stemmed tokens of the query (may contain duplicates).
    global_counts : dict[str, int]
        Mapping from every stemmed token in the corpus to its raw
        occurrence count across the entire corpus.
    total_corpus_tokens : int
        Sum of all token occurrences in the corpus
        (i.e. ``sum(global_counts.values())``).
    laplace_alpha : float
        Smoothing parameter for the corpus distribution.  The default
        Add-1 (Laplace) smoothing uses ``alpha = 1``.

    Returns
    -------
    dict with keys ``'ce'``, ``'kld'``, ``'jsd'`` (all floats).

    Notes
    -----
    * **P(w|Q)** -- frequency of token *w* in the query divided by the
      total number of query tokens.
    * **P(w|C)** -- global frequency of token *w* in the corpus divided
      by the total number of corpus tokens.  Laplace smoothing with
      parameter *laplace_alpha* is applied so that query words absent
      from the corpus still receive a non-zero probability.
    * CE   = -Σ P(w|Q) · log₂ P(w|C)
    * KLD  =  Σ P(w|Q) · log₂( P(w|Q) / P(w|C) )
    * JSD  = 0.5·KL(P||M) + 0.5·KL(Q||M)  where M = 0.5·(P+Q),
      computed in base 2.
    """
    if not query_tokens:
        return {"ce": 0.0, "kld": 0.0, "jsd": 0.0}

    # -- Build P(w|Q) ------------------------------------------------
    token_freq = {}
    for t in query_tokens:
        token_freq[t] = token_freq.get(t, 0) + 1
    unique_tokens = list(token_freq.keys())
    n_query = len(query_tokens)

    p_q = np.array([token_freq[t] / n_query for t in unique_tokens],
                    dtype=np.float64)

    # -- Build P(w|C) with Laplace smoothing --------------------------
    vocab_size = len(global_counts)
    smoothed_total = total_corpus_tokens + laplace_alpha * vocab_size
    p_c = np.array(
        [(global_counts.get(t, 0) + laplace_alpha) / smoothed_total
         for t in unique_tokens],
        dtype=np.float64,
    )

    # -- Cross-Entropy ------------------------------------------------
    ce = -np.sum(p_q * np.log2(p_c))

    # -- KL Divergence (P(w|Q) || P(w|C)) ----------------------------
    kld = np.sum(p_q * np.log2(p_q / p_c))

    # -- Jensen-Shannon Divergence (base 2) ---------------------------
    m = 0.5 * (p_q + p_c)
    kl_q_m = np.sum(p_q * np.log2(p_q / m))
    kl_c_m = np.sum(p_c * np.log2(p_c / m))
    jsd = 0.5 * kl_q_m + 0.5 * kl_c_m

    return {"ce": float(ce), "kld": float(kld), "jsd": float(jsd)}


def normalize_divergence_alphas(raw_pairs, apply_sigmoid, sig_center, sig_slope):
    """Min-max normalise a list of (query_id, raw_value) pairs to [0, 1].

    If *apply_sigmoid* is True the normalised value is further
    passed through a sigmoid.
    """
    values = [v for _, v in raw_pairs]
    lo, hi = min(values), max(values)
    spread = hi - lo if hi != lo else 1.0
    result = {}
    for qid, v in raw_pairs:
        normed = (v - lo) / spread
        if apply_sigmoid:
            normed = sigmoid(normed, sig_center, sig_slope)
        result[qid] = normed
    return result


# ============================================================
# 1. Fusion helpers
# ============================================================

def apply_rrf_fusion(bm25_results, dense_results, k=60):
    """Reciprocal Rank Fusion (Cormack et al. 2009).

    For each query, the document score is
      1 / (k + rank_BM25) + 1 / (k + rank_Dense)

    Returns {query_id: {doc_id: fused_score}}.
    """
    fused = {}
    all_qids = set(bm25_results.keys()) | set(dense_results.keys())
    for qid in all_qids:
        doc_scores = {}
        for rank, (doc_id, _) in enumerate(bm25_results.get(qid, []), start=1):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        for rank, (doc_id, _) in enumerate(dense_results.get(qid, []), start=1):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        fused[qid] = doc_scores
    return fused


def apply_wrrf_fusion(bm25_results, dense_results, alpha_map, k=60):
    """Weighted RRF: alpha * BM25_rrf + (1 - alpha) * Dense_rrf.

    *alpha_map* is {query_id: float in [0,1]}.
    """
    fused = {}
    all_qids = set(bm25_results.keys()) | set(dense_results.keys())
    for qid in all_qids:
        alpha = alpha_map.get(qid, 0.5)
        doc_scores = {}
        for rank, (doc_id, _) in enumerate(bm25_results.get(qid, []), start=1):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + alpha / (k + rank)
        for rank, (doc_id, _) in enumerate(dense_results.get(qid, []), start=1):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 - alpha) / (k + rank)
        fused[qid] = doc_scores
    return fused


def get_sorted_docs(scores_dict, top_k):
    """Sort *scores_dict* {doc_id: score} descending, return top-k as list of tuples."""
    return sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ============================================================
# 2. NDCG evaluation
# ============================================================

def calculate_ndcg_at_k(fused_results, qrels, top_k):
    """Compute average NDCG@k across all queries in *qrels*.

    *fused_results* is {query_id: {doc_id: score}}.
    """
    ndcg_scores = []
    for qid, rels in qrels.items():
        if qid not in fused_results:
            ndcg_scores.append(0.0)
            continue
        ranked = get_sorted_docs(fused_results[qid], top_k)
        dcg = 0.0
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            rel = rels.get(doc_id, 0)
            dcg += rel / math.log2(rank + 1)
        # Ideal DCG
        ideal_rels = sorted(rels.values(), reverse=True)[:top_k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


# ============================================================
# 3. Preprocessing (stemming / tokenization for BM25)
# ============================================================

def preprocess_corpus(corpus_jsonl, output_jsonl, stemmer_lang, batch_size=512):
    """Read a raw corpus JSONL, stem-tokenize every document, and write
    a tokenized version to *output_jsonl*.

    Each output line is {"_id": ..., "tokens": [...]}.
    Uses multiprocessing to parallelise stemming across CPU cores.

    Futures are drained in FIFO order via a bounded deque so that:
      - output document order matches the input (reproducible BM25 indices)
      - memory stays bounded even for multi-million-document corpora
    """
    ensure_dir(os.path.dirname(output_jsonl))
    n_workers = max(1, _N_CORES - 1)
    max_pending = n_workers * 3          # bound in-flight futures
    total_batches = math.ceil(count_lines(corpus_jsonl) / batch_size)

    with open(output_jsonl, "w", encoding="utf-8") as out:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(stemmer_lang,),
        ) as pool:
            pending = deque()
            pbar = tqdm(total=total_batches,
                        desc="  Preprocessing corpus", dynamic_ncols=True)

            for batch_ids, batch_texts in load_corpus_batch_generator(
                corpus_jsonl, batch_size
            ):
                pending.append(
                    pool.submit(stem_batch_worker, (batch_ids, batch_texts))
                )
                # Drain head of queue when buffer is full (FIFO order)
                while len(pending) >= max_pending:
                    for line in pending.popleft().result():
                        out.write(line + "\n")
                    pbar.update(1)

            # Flush remaining futures
            while pending:
                for line in pending.popleft().result():
                    out.write(line + "\n")
                pbar.update(1)

            pbar.close()


# ============================================================
# 4. BM25 index construction
# ============================================================

def build_bm25_and_word_freq_index(tokenized_corpus_jsonl):
    """Build BM25 index and corpus-wide word frequency index in a
    single pass over the tokenized corpus.

    Returns
    -------
    bm25 : BM25Okapi
        The BM25 index ready for querying.
    doc_ids : list[str]
        Ordered document IDs matching the BM25 index.
    global_counts : dict[str, int]
        Mapping from each stemmed token to its total occurrence count.
    total_tokens : int
        Sum of all token occurrences (``sum(global_counts.values())``).
    """
    doc_ids = []
    tokenized_docs = []
    global_counts = {}
    total_tokens = 0
    num_lines = count_lines(tokenized_corpus_jsonl)
    with open(tokenized_corpus_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="  Loading tokenized corpus",
                         dynamic_ncols=True):
            d = json.loads(line)
            doc_ids.append(d["_id"])
            tokens = d["tokens"]
            tokenized_docs.append(tokens)
            for t in tokens:
                global_counts[t] = global_counts.get(t, 0) + 1
            total_tokens += len(tokens)
    print(f"  Building BM25 index over {len(doc_ids):,} documents ...")
    bm25 = BM25Okapi(tokenized_docs)
    print(f"  Vocabulary size: {len(global_counts):,} unique tokens, "
          f"{total_tokens:,} total occurrences.")
    return bm25, doc_ids, global_counts, total_tokens


# ============================================================
# 5. BM25 retrieval
# ============================================================

def run_bm25_retrieval(bm25, doc_ids, queries, stemmer_lang, top_k):
    """Run BM25 queries and return {query_id: [(doc_id, score), ...]}."""
    stemmer = SnowballStemmer(stemmer_lang)
    results = {}
    for qid, qtext in tqdm(queries.items(), desc="  BM25 retrieval",
                            dynamic_ncols=True):
        tokens = stem_and_tokenize(qtext, stemmer)
        scores = bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        results[qid] = [(doc_ids[i], float(scores[i])) for i in top_idx]
    return results


# ============================================================
# 6. Dense embeddings
# ============================================================

def _encode_with_oom_retry(model, texts, device, batch_size):
    """Encode *texts* on *device*, halving *batch_size* on CUDA OOM.

    Returns a list of CPU tensors (one per successful sub-batch).
    """
    results = []
    start = 0
    current_bs = batch_size
    while start < len(texts):
        end = min(start + current_bs, len(texts))
        sub_batch = texts[start:end]
        try:
            embs = model.encode(
                sub_batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=device,
            )
            results.append(embs.cpu())
            start = end
            # Restore original batch size for the next chunk
            current_bs = batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            current_bs = max(1, current_bs // 2)
            if current_bs < batch_size:
                print(f"\n  [OOM] Reduced sub-batch size to {current_bs}")
            if current_bs == 1 and end - start == 1:
                # Single item still OOMs — re-raise
                raise
    return results


def build_corpus_embeddings(corpus_jsonl, model, batch_size, device):
    """Encode every document in *corpus_jsonl* and return a stacked
    tensor of embeddings plus the ordered list of doc IDs.
    """
    all_ids, all_embs = [], []
    total = count_lines(corpus_jsonl)
    for batch_ids, batch_texts in tqdm(
        load_corpus_batch_generator(corpus_jsonl, batch_size),
        total=math.ceil(total / batch_size),
        desc="  Encoding corpus",
        dynamic_ncols=True,
    ):
        embs_list = _encode_with_oom_retry(model, batch_texts, device, batch_size)
        all_ids.extend(batch_ids)
        all_embs.extend(embs_list)
    return torch.cat(all_embs, dim=0), all_ids


# ============================================================
# 7. Dense retrieval
# ============================================================

def build_dense_query_vectors(queries, model, batch_size, device):
    """Encode all queries and return a tensor plus the ordered query IDs."""
    qids = list(queries.keys())
    qtexts = [queries[q] for q in qids]
    all_embs = []
    for i in tqdm(range(0, len(qtexts), batch_size), desc="  Encoding queries",
                  dynamic_ncols=True):
        batch = qtexts[i : i + batch_size]
        embs_list = _encode_with_oom_retry(model, batch, device, batch_size)
        all_embs.extend(embs_list)
    return torch.cat(all_embs, dim=0), qids


def run_dense_retrieval(
    query_vectors, query_ids, corpus_vectors, corpus_ids, top_k,
    corpus_chunk_size, device, query_chunk_size=100,
):
    """Cosine-similarity search over pre-computed embeddings.

    Query chunks are moved to *device* before calling semantic_search so
    that the dot-product runs on the GPU when one is available.  The
    corpus tensor stays on CPU; sentence-transformers transfers corpus
    chunks to the same device as the query internally.

    Returns {query_id: [(doc_id, score), :]}.
    """
    results = {}
    n_queries = len(query_ids)
    # Process queries in chunks to limit memory usage
    for q_start in tqdm(
        range(0, n_queries, query_chunk_size), desc="  Dense retrieval",
        dynamic_ncols=True,
    ):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_batch = query_vectors[q_start:q_end].to(device)
        hits = st_util.semantic_search(
            q_batch, corpus_vectors, top_k=top_k,
            corpus_chunk_size=corpus_chunk_size,
        )
        for idx, hit_list in enumerate(hits):
            qid = query_ids[q_start + idx]
            results[qid] = [
                (corpus_ids[h["corpus_id"]], float(h["score"])) for h in hit_list
            ]
    return results


# ============================================================
# 8. Benchmark evaluation -- run all 9 methods
# ============================================================

def evaluate_all_methods(
    bm25_results, dense_results, qrels, cfg_benchmark,
    queries, stemmer_lang, global_counts, total_corpus_tokens,
):
    """Run every fusion strategy and return a list of (method_name, ndcg_score).

    The divergence-based alpha is computed from **word-level**
    distributions (P(w|Q) vs P(w|C)), not from retrieval scores.

    Parameters
    ----------
    queries : dict[str, str]
        Raw query texts keyed by query id (used to build P(w|Q)).
    stemmer_lang : str
        Language for the Snowball stemmer.
    global_counts : dict[str, int]
        Corpus-wide token frequency index.
    total_corpus_tokens : int
        Total token occurrences in the corpus.
    """
    ndcg_k = cfg_benchmark["ndcg_k"]
    rrf_k = cfg_benchmark["rrf"]["k"]
    sig_center = cfg_benchmark["sigmoid"]["center"]
    sig_slope = cfg_benchmark["sigmoid"]["slope"]

    scores = []

    # -- 1. BM25 Only -----------------------------------------------
    bm25_fused = {
        qid: {doc: s for doc, s in docs}
        for qid, docs in bm25_results.items()
    }
    scores.append(("BM25 Only", calculate_ndcg_at_k(bm25_fused, qrels, ndcg_k)))

    # -- 2. Dense Only ----------------------------------------------
    dense_fused = {
        qid: {doc: s for doc, s in docs}
        for qid, docs in dense_results.items()
    }
    scores.append(("Dense Only", calculate_ndcg_at_k(dense_fused, qrels, ndcg_k)))

    # -- 3. Naive RRF -----------------------------------------------
    rrf_fused = apply_rrf_fusion(bm25_results, dense_results, k=rrf_k)
    scores.append(("Naive RRF", calculate_ndcg_at_k(rrf_fused, qrels, ndcg_k)))

    laplace_alpha = cfg_benchmark.get("smoothing", {}).get("laplace_alpha", 1)

    # -- Pre-compute per-query metrics from word distributions ------
    stemmer = SnowballStemmer(stemmer_lang)
    query_metrics = {}  # qid -> {'ce': float, 'kld': float, 'jsd': float}
    for qid, qtext in queries.items():
        tokens = stem_and_tokenize(qtext, stemmer)
        query_metrics[qid] = compute_query_metrics(
            tokens, global_counts, total_corpus_tokens, laplace_alpha
        )

    # Divergence-based weighted RRF for the remaining 6 methods.
    # Each mode picks one of the three raw metrics: jsd, kld, or ce.
    divergence_modes = [
        ("JSD (Linear)",                  "jsd",  False, False),
        ("JSD (Sigmoid)",                 "jsd",  False, True),
        ("KLD (0-1 Norm)",                "kld",  True,  False),
        ("KLD (0-1 + Sigmoid)",           "kld",  True,  True),
        ("Cross-Entropy (0-1 Norm)",      "ce",   True,  False),
        ("Cross-Entropy (0-1 + Sigmoid)", "ce",   True,  True),
    ]

    for method_label, metric_key, needs_norm, needs_sigmoid in divergence_modes:
        raw_pairs = [(qid, query_metrics[qid][metric_key]) for qid in qrels]

        if needs_norm:
            alpha_map = normalize_divergence_alphas(
                raw_pairs, needs_sigmoid, sig_center, sig_slope
            )
        elif needs_sigmoid:
            alpha_map = {
                qid: sigmoid(val, sig_center, sig_slope)
                for qid, val in raw_pairs
            }
        else:
            # Linear (raw value used directly as alpha)
            alpha_map = {qid: val for qid, val in raw_pairs}

        wrrf_fused = apply_wrrf_fusion(bm25_results, dense_results, alpha_map, k=rrf_k)
        ndcg = calculate_ndcg_at_k(wrrf_fused, qrels, ndcg_k)
        scores.append((method_label, ndcg))

    return scores


# ============================================================
# 9. Top-level pipeline orchestration
# ============================================================


def run_pipeline_for_dataset(dataset_name, cfg, model, device):
    """Execute every pipeline step for a single dataset.

    Cached artifacts are stored under
      <results_folder>/<model_short>/<dataset_name>/
    so repeated runs skip completed steps automatically.
    """
    short_model = model_short_name(cfg["embeddings"]["model_name"])
    ds_dir = os.path.join(
        cfg["paths"]["results_folder"], short_model, dataset_name
    )
    ensure_dir(ds_dir)

    datasets_folder = cfg["paths"]["datasets_folder"]
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    emb_batch_size = cfg["embeddings"]["batch_size"]
    top_k = cfg["benchmark"]["top_k"]
    dense_cfg = cfg["dense_search"]

    # Paths for intermediate artifacts
    corpus_jsonl = os.path.join(ds_dir, "corpus.jsonl")
    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
    qrels_tsv = os.path.join(ds_dir, "qrels.tsv")
    tokenized_jsonl = os.path.join(ds_dir, "tokenized_corpus.jsonl")
    bm25_pkl = os.path.join(ds_dir, "bm25_index.pkl")
    bm25_docids_pkl = os.path.join(ds_dir, "bm25_doc_ids.pkl")
    corpus_emb_pt = os.path.join(ds_dir, "corpus_embeddings.pt")
    corpus_ids_pkl = os.path.join(ds_dir, "corpus_ids.pkl")
    bm25_results_pkl = os.path.join(ds_dir, "bm25_results.pkl")
    word_freq_pkl = os.path.join(ds_dir, "word_freq_index.pkl")
    query_vectors_pt = os.path.join(ds_dir, "query_vectors.pt")
    query_ids_pkl = os.path.join(ds_dir, "query_ids.pkl")
    dense_results_pkl = os.path.join(ds_dir, "dense_results.pkl")
    results_csv = os.path.join(ds_dir, "results.csv")
    results_png = os.path.join(ds_dir, "results.png")

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Output : {ds_dir}")
    print(f"{'=' * 60}")

    timings = []           # (step_label, elapsed_seconds)
    timing_csv = os.path.join(ds_dir, "timing.csv")

    # ----------------------------------------------------------
    # Step 1 -- Download the BEIR dataset
    # ----------------------------------------------------------
    print("\n[Step 1/10] Downloading / verifying dataset ...")
    t0 = time.time()
    dataset_path = download_beir_dataset(dataset_name, datasets_folder)
    timings.append(("Step 1: Download / verify dataset", time.time() - t0))
    if dataset_path is None:
        print("  SKIPPED (download failed). Aborting this dataset.")
        return

    # ----------------------------------------------------------
    # Step 2 -- Load dataset and write local copies of corpus / queries / qrels
    # ----------------------------------------------------------
    t0 = time.time()
    if file_exists(corpus_jsonl) and file_exists(queries_jsonl) and file_exists(qrels_tsv):
        print("[Step 2/10] Corpus / queries / qrels already cached. Loading ...")
        queries = load_queries(queries_jsonl)
        qrels = load_qrels(qrels_tsv)
    else:
        print("[Step 2/10] Loading BEIR data and writing local copies ...")
        corpus, queries, qrels, split = load_beir_dataset(dataset_path)
        if corpus is None:
            print("  SKIPPED (failed to load). Aborting this dataset.")
            return
        print(f"  Split: {split} | Corpus: {len(corpus):,} | Queries: {len(queries):,}")
        write_corpus_jsonl(corpus, corpus_jsonl)
        write_queries_jsonl(queries, queries_jsonl)
        write_qrels_tsv(qrels, qrels_tsv)
        del corpus  # free memory
    timings.append(("Step 2: Load / write corpus, queries, qrels", time.time() - t0))

    # ----------------------------------------------------------
    # Step 3 -- Preprocess (stem + tokenize) the corpus for BM25
    # ----------------------------------------------------------
    t0 = time.time()
    if file_exists(tokenized_jsonl):
        print("[Step 3/10] Tokenized corpus already exists. Skipping.")
    else:
        print("[Step 3/10] Preprocessing corpus (stemming + tokenization) ...")
        preprocess_corpus(corpus_jsonl, tokenized_jsonl, stemmer_lang)
    timings.append(("Step 3: Preprocessing (stem + tokenize)", time.time() - t0))

    # ----------------------------------------------------------
    # Step 4 -- Build the BM25 index and word frequency index
    # ----------------------------------------------------------
    t0 = time.time()
    all_indices_cached = (
        file_exists(bm25_pkl) and file_exists(bm25_docids_pkl)
        and file_exists(word_freq_pkl)
    )
    if all_indices_cached:
        print("[Step 4/10] BM25 & word-freq indices already exist. Loading ...")
        bm25 = load_pickle(bm25_pkl)
        bm25_doc_ids = load_pickle(bm25_docids_pkl)
        global_counts, total_corpus_tokens = load_pickle(word_freq_pkl)
    else:
        print("[Step 4/10] Building BM25 index & word frequency index ...")
        bm25, bm25_doc_ids, global_counts, total_corpus_tokens = (
            build_bm25_and_word_freq_index(tokenized_jsonl)
        )
        save_pickle(bm25, bm25_pkl)
        save_pickle(bm25_doc_ids, bm25_docids_pkl)
        save_pickle((global_counts, total_corpus_tokens), word_freq_pkl)
    timings.append(("Step 4: BM25 + word-freq index", time.time() - t0))

    # ----------------------------------------------------------
    # Step 5 -- Encode the corpus with the embedding model
    # ----------------------------------------------------------
    t0 = time.time()
    if file_exists(corpus_emb_pt) and file_exists(corpus_ids_pkl):
        print("[Step 5/10] Corpus embeddings already exist. Loading ...")
        corpus_embeddings = torch.load(corpus_emb_pt, weights_only=True)
        corpus_ids = load_pickle(corpus_ids_pkl)
    else:
        print("[Step 5/10] Encoding corpus with embedding model ...")
        corpus_embeddings, corpus_ids = build_corpus_embeddings(
            corpus_jsonl, model, emb_batch_size, device
        )
        torch.save(corpus_embeddings, corpus_emb_pt)
        save_pickle(corpus_ids, corpus_ids_pkl)
    timings.append(("Step 5: Corpus embeddings", time.time() - t0))

    # ----------------------------------------------------------
    # Step 6 -- BM25 retrieval
    # ----------------------------------------------------------
    t0 = time.time()
    if file_exists(bm25_results_pkl):
        print("[Step 6/10] BM25 results already exist. Loading ...")
        bm25_results = load_pickle(bm25_results_pkl)
    else:
        print("[Step 6/10] Running BM25 retrieval ...")
        if not queries:
            queries = load_queries(queries_jsonl)
        bm25_results = run_bm25_retrieval(
            bm25, bm25_doc_ids, queries, stemmer_lang, top_k
        )
        save_pickle(bm25_results, bm25_results_pkl)
    timings.append(("Step 6: BM25 retrieval", time.time() - t0))

    # Free the BM25 index -- it is no longer needed
    del bm25, bm25_doc_ids

    # ----------------------------------------------------------
    # Step 7 -- Encode queries with the embedding model
    # ----------------------------------------------------------
    t0 = time.time()
    if file_exists(query_vectors_pt) and file_exists(query_ids_pkl):
        print("[Step 7/10] Query vectors already exist. Loading ...")
        query_vectors = torch.load(query_vectors_pt, weights_only=True)
        query_ids = load_pickle(query_ids_pkl)
    else:
        print("[Step 7/10] Encoding queries ...")
        if not queries:
            queries = load_queries(queries_jsonl)
        query_vectors, query_ids = build_dense_query_vectors(
            queries, model, emb_batch_size, device
        )
        torch.save(query_vectors, query_vectors_pt)
        save_pickle(query_ids, query_ids_pkl)
    timings.append(("Step 7: Query embeddings", time.time() - t0))

    # ----------------------------------------------------------
    # Step 8 -- Dense retrieval
    # ----------------------------------------------------------
    t0 = time.time()
    if file_exists(dense_results_pkl):
        print("[Step 8/10] Dense results already exist. Loading ...")
        dense_results = load_pickle(dense_results_pkl)
    else:
        print("[Step 8/10] Running dense retrieval ...")
        dense_results = run_dense_retrieval(
            query_vectors, query_ids, corpus_embeddings, corpus_ids, top_k,
            dense_cfg["corpus_chunk_size"],
            device=device,
            query_chunk_size=dense_cfg.get("query_chunk_size", 100),
        )
        save_pickle(dense_results, dense_results_pkl)
    timings.append(("Step 8: Dense retrieval", time.time() - t0))

    # Free embeddings
    del corpus_embeddings, query_vectors

    # ----------------------------------------------------------
    # Step 9 -- Evaluate all fusion methods
    # ----------------------------------------------------------
    print("[Step 9/10] Evaluating fusion methods ...")
    t0 = time.time()
    if not qrels:
        qrels = load_qrels(qrels_tsv)
    if not queries:
        queries = load_queries(queries_jsonl)
    method_scores = evaluate_all_methods(
        bm25_results, dense_results, qrels, cfg["benchmark"],
        queries, stemmer_lang, global_counts, total_corpus_tokens,
    )
    timings.append(("Step 9: Evaluation (all fusion methods)", time.time() - t0))

    # Print a quick summary table to stdout
    print(f"\n  {'Method':<32s} {'NDCG@10':>8s}")
    print(f"  {'-'*32} {'-'*8}")
    for method, ndcg in method_scores:
        print(f"  {method:<32s} {ndcg:>8.4f}")

    # ----------------------------------------------------------
    # Step 10 -- Save results (CSV + chart + timing)
    # ----------------------------------------------------------
    print("\n[Step 10/10] Saving results ...")
    t0 = time.time()
    save_results_csv(method_scores, results_csv)
    print(f"  Results saved to {results_csv}")
    save_results_chart(
        method_scores, results_png, dataset_name, short_model
    )
    timings.append(("Step 10: Save results", time.time() - t0))

    # Write timing stats
    save_timing_csv(timings, timing_csv)
    print(f"  Timing saved to {timing_csv}")

    # Print timing summary
    print(f"\n  {'Step':<45s} {'Time (s)':>10s}")
    print(f"  {'-'*45} {'-'*10}")
    for label, secs in timings:
        print(f"  {label:<45s} {secs:>10.2f}")
    total_time = sum(t for _, t in timings)
    print(f"  {'-'*45} {'-'*10}")
    print(f"  {'Total':<45s} {total_time:>10.2f}")

    print(f"\n  Finished {dataset_name}.")


def run_pipeline_merged(datasets, cfg, model, device):
    """Merge all specified datasets into one corpus, then run the
    benchmark once over the combined data.

    Merged artifacts go to  results/<model>/merged/
    """
    short_model = model_short_name(cfg["embeddings"]["model_name"])
    merged_dir = os.path.join(
        cfg["paths"]["results_folder"], short_model, "merged"
    )
    ensure_dir(merged_dir)
    datasets_folder = cfg["paths"]["datasets_folder"]

    corpus_jsonl = os.path.join(merged_dir, "corpus.jsonl")
    queries_jsonl = os.path.join(merged_dir, "queries.jsonl")
    qrels_tsv = os.path.join(merged_dir, "qrels.tsv")

    print(f"\n{'=' * 60}")
    print(f"  MERGE MODE -- combining {len(datasets)} datasets")
    print(f"  Output : {merged_dir}")
    print(f"{'=' * 60}")

    # ----------------------------------------------------------
    # Download + merge all datasets
    # ----------------------------------------------------------
    if file_exists(corpus_jsonl) and file_exists(queries_jsonl) and file_exists(qrels_tsv):
        print("\n[Merge] Merged files already exist. Skipping merge step.")
    else:
        initialize_output_files(corpus_jsonl, queries_jsonl, qrels_tsv)
        for ds_name in tqdm(datasets, desc="  Downloading & merging",
                            dynamic_ncols=True):
            ds_path = download_beir_dataset(ds_name, datasets_folder)
            if ds_path is None:
                continue
            corpus, queries, qrels, split = load_beir_dataset(ds_path)
            if corpus is None:
                continue
            print(f"    {ds_name}: {len(corpus):,} docs, {len(queries):,} queries (split: {split})")
            append_corpus_to_jsonl(corpus, corpus_jsonl, ds_name)
            append_queries_to_jsonl(queries, queries_jsonl, ds_name)
            append_qrels_to_tsv(qrels, qrels_tsv, ds_name)
            del corpus, queries, qrels

    # Now hand off to the normal per-dataset pipeline which will
    # pick up the already-written corpus/queries/qrels.
    # We create a temporary config override so it reads from merged_dir.
    _run_benchmark_steps(merged_dir, cfg, model, device, label="merged")


def _run_benchmark_steps(ds_dir, cfg, model, device, label="dataset"):
    """Execute steps 3-10 (preprocess through evaluation) given that
    corpus.jsonl, queries.jsonl, and qrels.tsv already exist in *ds_dir*.

    This is shared by both the per-dataset and merged code paths.
    """
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    emb_batch_size = cfg["embeddings"]["batch_size"]
    top_k = cfg["benchmark"]["top_k"]
    dense_cfg = cfg["dense_search"]
    short_model = model_short_name(cfg["embeddings"]["model_name"])

    corpus_jsonl = os.path.join(ds_dir, "corpus.jsonl")
    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
    qrels_tsv = os.path.join(ds_dir, "qrels.tsv")
    tokenized_jsonl = os.path.join(ds_dir, "tokenized_corpus.jsonl")
    bm25_pkl = os.path.join(ds_dir, "bm25_index.pkl")
    bm25_docids_pkl = os.path.join(ds_dir, "bm25_doc_ids.pkl")
    corpus_emb_pt = os.path.join(ds_dir, "corpus_embeddings.pt")
    corpus_ids_pkl = os.path.join(ds_dir, "corpus_ids.pkl")
    bm25_results_pkl = os.path.join(ds_dir, "bm25_results.pkl")
    word_freq_pkl = os.path.join(ds_dir, "word_freq_index.pkl")
    query_vectors_pt = os.path.join(ds_dir, "query_vectors.pt")
    query_ids_pkl = os.path.join(ds_dir, "query_ids.pkl")
    dense_results_pkl = os.path.join(ds_dir, "dense_results.pkl")
    results_csv = os.path.join(ds_dir, "results.csv")
    results_png = os.path.join(ds_dir, "results.png")

    timings = []           # (step_label, elapsed_seconds)
    timing_csv = os.path.join(ds_dir, "timing.csv")

    queries = load_queries(queries_jsonl)
    qrels = load_qrels(qrels_tsv)

    # Preprocessing
    t0 = time.time()
    if file_exists(tokenized_jsonl):
        print("[Step 3] Tokenized corpus exists. Skipping.")
    else:
        print("[Step 3] Preprocessing corpus ...")
        preprocess_corpus(corpus_jsonl, tokenized_jsonl, stemmer_lang)
    timings.append(("Step 3: Preprocessing (stem + tokenize)", time.time() - t0))

    # BM25 index + word frequency index (single pass)
    t0 = time.time()
    all_indices_cached = (
        file_exists(bm25_pkl) and file_exists(bm25_docids_pkl)
        and file_exists(word_freq_pkl)
    )
    if all_indices_cached:
        print("[Step 4] BM25 & word-freq indices exist. Loading ...")
        bm25 = load_pickle(bm25_pkl)
        bm25_doc_ids = load_pickle(bm25_docids_pkl)
        global_counts, total_corpus_tokens = load_pickle(word_freq_pkl)
    else:
        print("[Step 4] Building BM25 index & word frequency index ...")
        bm25, bm25_doc_ids, global_counts, total_corpus_tokens = (
            build_bm25_and_word_freq_index(tokenized_jsonl)
        )
        save_pickle(bm25, bm25_pkl)
        save_pickle(bm25_doc_ids, bm25_docids_pkl)
        save_pickle((global_counts, total_corpus_tokens), word_freq_pkl)
    timings.append(("Step 4: BM25 + word-freq index", time.time() - t0))

    # Corpus embeddings
    t0 = time.time()
    if file_exists(corpus_emb_pt) and file_exists(corpus_ids_pkl):
        print("[Step 5] Corpus embeddings exist. Loading ...")
        corpus_embeddings = torch.load(corpus_emb_pt, weights_only=True)
        corpus_ids = load_pickle(corpus_ids_pkl)
    else:
        print("[Step 5] Encoding corpus ...")
        corpus_embeddings, corpus_ids = build_corpus_embeddings(
            corpus_jsonl, model, emb_batch_size, device
        )
        torch.save(corpus_embeddings, corpus_emb_pt)
        save_pickle(corpus_ids, corpus_ids_pkl)
    timings.append(("Step 5: Corpus embeddings", time.time() - t0))

    # BM25 retrieval
    t0 = time.time()
    if file_exists(bm25_results_pkl):
        print("[Step 6] BM25 results exist. Loading ...")
        bm25_results = load_pickle(bm25_results_pkl)
    else:
        print("[Step 6] Running BM25 retrieval ...")
        bm25_results = run_bm25_retrieval(bm25, bm25_doc_ids, queries, stemmer_lang, top_k)
        save_pickle(bm25_results, bm25_results_pkl)
    timings.append(("Step 6: BM25 retrieval", time.time() - t0))

    del bm25, bm25_doc_ids

    # Query embeddings
    t0 = time.time()
    if file_exists(query_vectors_pt) and file_exists(query_ids_pkl):
        print("[Step 7] Query vectors exist. Loading ...")
        query_vectors = torch.load(query_vectors_pt, weights_only=True)
        query_ids = load_pickle(query_ids_pkl)
    else:
        print("[Step 7] Encoding queries ...")
        query_vectors, query_ids = build_dense_query_vectors(
            queries, model, emb_batch_size, device
        )
        torch.save(query_vectors, query_vectors_pt)
        save_pickle(query_ids, query_ids_pkl)
    timings.append(("Step 7: Query embeddings", time.time() - t0))

    # Dense retrieval
    t0 = time.time()
    if file_exists(dense_results_pkl):
        print("[Step 8] Dense results exist. Loading ...")
        dense_results = load_pickle(dense_results_pkl)
    else:
        print("[Step 8] Running dense retrieval ...")
        dense_results = run_dense_retrieval(
            query_vectors, query_ids, corpus_embeddings, corpus_ids, top_k,
            dense_cfg["corpus_chunk_size"],
            device=device,
            query_chunk_size=dense_cfg.get("query_chunk_size", 100),
        )
        save_pickle(dense_results, dense_results_pkl)
    timings.append(("Step 8: Dense retrieval", time.time() - t0))

    del corpus_embeddings, query_vectors

    # Evaluation
    print("[Step 9] Evaluating fusion methods ...")
    t0 = time.time()
    method_scores = evaluate_all_methods(
        bm25_results, dense_results, qrels, cfg["benchmark"],
        queries, stemmer_lang, global_counts, total_corpus_tokens,
    )
    timings.append(("Step 9: Evaluation (all fusion methods)", time.time() - t0))

    print(f"\n  {'Method':<32s} {'NDCG@10':>8s}")
    print(f"  {'-'*32} {'-'*8}")
    for method, ndcg in method_scores:
        print(f"  {method:<32s} {ndcg:>8.4f}")

    # Save outputs
    print("\n[Step 10] Saving results ...")
    t0 = time.time()
    save_results_csv(method_scores, results_csv)
    print(f"  Results saved to {results_csv}")
    save_results_chart(method_scores, results_png, label, short_model)
    timings.append(("Step 10: Save results", time.time() - t0))

    # Write timing stats
    save_timing_csv(timings, timing_csv)
    print(f"  Timing saved to {timing_csv}")

    # Print timing summary
    print(f"\n  {'Step':<45s} {'Time (s)':>10s}")
    print(f"  {'-'*45} {'-'*10}")
    for label_t, secs in timings:
        print(f"  {label_t:<45s} {secs:>10.2f}")
    total_time = sum(t for _, t in timings)
    print(f"  {'-'*45} {'-'*10}")
    print(f"  {'Total':<45s} {total_time:>10.2f}")

    print(f"\n  Finished {label}.")


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG-LLM Hybrid Retrieval Benchmark Pipeline"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    # Override the module-level config path if the user supplied one
    import src.utils as _u
    _u.CONFIG_PATH = args.config

    cfg = load_config()

    datasets = cfg.get("datasets", [])
    if not datasets:
        print("No datasets specified in config.yaml. Nothing to do.")
        sys.exit(0)

    merge_mode = cfg.get("merge", False)
    model_name = cfg["embeddings"]["model_name"]

    # Select compute device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device : {device}")
    print(f"Model  : {model_name}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"Mode   : {'MERGE' if merge_mode else 'PER-DATASET'}")

    # Load the sentence-transformer model once.
    # HF download bars are suppressed (env var set above) to prevent
    # overlapping with our own progress indicators.
    print("\nLoading embedding model ...")
    start = time.time()
    model = SentenceTransformer(model_name, device=device)
    elapsed = time.time() - start
    print(f"  Model loaded in {elapsed:.1f}s")
    print(f"  CPU threads (torch)     : {torch.get_num_threads()}")
    print(f"  CPU inter-op threads    : {torch.get_num_interop_threads()}")
    print(f"  Workers (preprocessing) : {max(1, _N_CORES - 1)}\n")

    if merge_mode:
        run_pipeline_merged(datasets, cfg, model, device)
    else:
        for ds_name in datasets:
            run_pipeline_for_dataset(ds_name, cfg, model, device)

    print("\n" + "=" * 60)
    print("  All done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
