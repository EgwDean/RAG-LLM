"""
utils.py -- Shared utilities for the RAG-LLM hybrid retrieval pipeline.

Contains helpers for:
  - Configuration loading
  - File / directory management
  - Pickle serialization
  - Data loaders (queries, qrels, corpus batches)
  - BEIR dataset download and loading
  - Corpus / query / qrel append routines for the merged mode
"""

import os
import csv
import json
import pickle
import yaml


# ============================================================
# Path to the project-level configuration file.
# The pipeline always runs from the project root, so a
# relative path is fine here.
# ============================================================
CONFIG_PATH = "config.yaml"


# ============================================================
# Configuration
# ============================================================

def load_config():
    """Read and return the YAML configuration dictionary.

    Raises FileNotFoundError if config.yaml is missing.
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Configuration file not found at: {os.path.abspath(CONFIG_PATH)}"
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_path(cfg, key, default_value):
    """Read a path key from cfg['paths'] with a fallback default."""
    paths_cfg = cfg.get("paths", {}) or {}
    return paths_cfg.get(key, default_value)


def _format_bm25_float(value):
    """Format BM25 float params for stable, human-readable cache keys."""
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def get_bm25_params(cfg, override=None):
    """Return BM25 params from config, optionally overridden by a dict."""
    bm25_cfg = dict(cfg.get("bm25", {}) or {})
    if override:
        bm25_cfg.update(override)

    k1 = float(bm25_cfg.get("k1", 1.5))
    b = float(bm25_cfg.get("b", 0.75))
    use_stemming = bool(bm25_cfg.get("use_stemming", True))
    return {"k1": k1, "b": b, "use_stemming": use_stemming}


def bm25_signature(k1, b, use_stemming):
    """Build a filename-safe signature for BM25 cache artifacts."""
    return (
        f"bm25_k1_{_format_bm25_float(k1)}"
        f"_b_{_format_bm25_float(b)}"
        f"_stem_{1 if use_stemming else 0}"
    )


def bm25_artifact_paths(ds_dir, k1, b, use_stemming, top_k=None):
    """Return sparse artifact paths for one dataset and BM25 config.

    BM25 retrieval-result caches are keyed by top_k to avoid stale cache reuse
    when benchmark.top_k changes.
    """
    sig = bm25_signature(k1, b, use_stemming)
    stem_flag = f"stem_{1 if use_stemming else 0}"
    if top_k is None:
        bm25_results_name = f"{sig}_results.pkl"
    else:
        bm25_results_name = f"{sig}_topk_{int(top_k)}_results.pkl"
    return {
        "tokenized_corpus_jsonl": os.path.join(ds_dir, f"tokenized_corpus_{stem_flag}.jsonl"),
        "tokenized_queries_jsonl": os.path.join(ds_dir, f"tokenized_queries_{stem_flag}.jsonl"),
        "query_tokens_pkl": os.path.join(ds_dir, f"query_tokens_{stem_flag}.pkl"),
        "word_freq_pkl": os.path.join(ds_dir, f"word_freq_index_{stem_flag}.pkl"),
        "doc_freq_pkl": os.path.join(ds_dir, f"doc_freq_index_{stem_flag}.pkl"),
        "bm25_pkl": os.path.join(ds_dir, f"{sig}.pkl"),
        "bm25_docids_pkl": os.path.join(ds_dir, f"{sig}_doc_ids.pkl"),
        "bm25_results_pkl": os.path.join(ds_dir, bm25_results_name),
        "bm25_signature": sig,
    }


# ============================================================
# File / directory helpers
# ============================================================

def ensure_dir(path):
    """Create *path* and all intermediate directories if they do not exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def count_lines(filepath):
    """Return the number of lines in *filepath* (binary mode for speed)."""
    count = 0
    with open(filepath, "rb") as f:
        for _ in f:
            count += 1
    return count


def file_exists(path):
    """Return True if *path* points to an existing file."""
    return os.path.isfile(path)


# ============================================================
# Pickle serialization
# ============================================================

def save_pickle(data, filepath):
    """Serialize *data* to *filepath*, creating parent directories as needed."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    """Deserialize and return the object stored in *filepath*."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ============================================================
# Data loaders
# ============================================================

def load_queries(filepath):
    """Load queries from a JSONL file.

    Returns a dict  {query_id: query_text}.
    """
    queries = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    return queries


def load_qrels(filepath):
    """Load relevance judgments from a TSV file.

    Returns a nested dict  {query_id: {doc_id: int_score}}.
    """
    qrels = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][row["corpus-id"]] = int(row["score"])
    return qrels


def load_corpus_batch_generator(filepath, batch_size):
    """Yield (doc_ids, texts) batches from a JSONL corpus file.

    Each document's title and text are concatenated into a single
    string before being returned.  Malformed JSON lines are silently
    skipped.
    """
    batch_ids, batch_texts = [], []
    malformed_lines = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)
                full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                batch_ids.append(doc["_id"])
                batch_texts.append(full_text)
                if len(batch_texts) >= batch_size:
                    yield batch_ids, batch_texts
                    batch_ids, batch_texts = [], []
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
    if batch_texts:
        yield batch_ids, batch_texts
    if malformed_lines > 0:
        print(
            f"  [WARN] Skipped {malformed_lines} malformed JSON lines in corpus file: {filepath}"
        )


# ============================================================
# BEIR dataset download / load helpers
# ============================================================

def download_beir_dataset(dataset_name, datasets_folder):
    """Download and unzip a BEIR dataset if not already present on disk.

    Returns the path to the extracted dataset directory, or None on
    failure.
    """
    from beir import util as beir_util  # imported here to keep startup fast

    dataset_path = os.path.join(datasets_folder, dataset_name)
    if os.path.exists(dataset_path):
        return dataset_path

    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/"
        f"thakur/BEIR/datasets/{dataset_name}.zip"
    )
    try:
        beir_util.download_and_unzip(url, datasets_folder)
        return dataset_path
    except Exception as exc:
        print(f"  [ERROR] Download failed for {dataset_name}: {exc}")
        return None


def load_beir_dataset(dataset_path):
    """Load a BEIR dataset from *dataset_path*.

    Automatically selects the best available split by checking
    for test -> dev -> train in the qrels directory.

    Returns (corpus, queries, qrels, split_name) or (None, ...) on
    failure.
    """
    from beir.datasets.data_loader import GenericDataLoader

    loader = GenericDataLoader(dataset_path)

    # Pick the first available split
    split = None
    for candidate in ["test", "dev", "train"]:
        qrel_file = os.path.join(dataset_path, "qrels", f"{candidate}.tsv")
        if os.path.exists(qrel_file):
            split = candidate
            break

    if split is None:
        print(f"  [ERROR] No valid qrels split found in {dataset_path}")
        return None, None, None, None

    try:
        corpus, queries, qrels = loader.load(split=split)
        return corpus, queries, qrels, split
    except Exception as exc:
        print(f"  [ERROR] Failed to load {dataset_path}: {exc}")
        return None, None, None, None


# ============================================================
# Merge helpers (used when merge=true)
# ============================================================

def initialize_output_files(corpus_path, queries_path, qrels_path):
    """Create empty output files and write the qrels TSV header.

    Called once before appending data from multiple datasets.
    """
    open(corpus_path, "w", encoding="utf-8").close()
    open(queries_path, "w", encoding="utf-8").close()
    ensure_dir(os.path.dirname(qrels_path))
    with open(qrels_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])


def append_corpus_to_jsonl(corpus_dict, filepath, dataset_prefix):
    """Append corpus documents to a JSONL file, prefixing IDs
    with *dataset_prefix* to avoid collisions across datasets.
    """
    if not corpus_dict:
        return
    with open(filepath, "a", encoding="utf-8") as f:
        for doc_id, doc in corpus_dict.items():
            entry = {
                "_id": f"{dataset_prefix}_{doc_id}",
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
            json.dump(entry, f)
            f.write("\n")


def append_queries_to_jsonl(queries_dict, filepath, dataset_prefix):
    """Append queries to a JSONL file, prefixing IDs."""
    if not queries_dict:
        return
    with open(filepath, "a", encoding="utf-8") as f:
        for q_id, q_text in queries_dict.items():
            entry = {"_id": f"{dataset_prefix}_{q_id}", "text": q_text}
            json.dump(entry, f)
            f.write("\n")


def append_qrels_to_tsv(qrels_dict, filepath, dataset_prefix):
    """Append relevance judgments to a TSV file, prefixing IDs."""
    if not qrels_dict:
        return
    with open(filepath, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for q_id, doc_map in qrels_dict.items():
            for doc_id, score in doc_map.items():
                writer.writerow([
                    f"{dataset_prefix}_{q_id}",
                    f"{dataset_prefix}_{doc_id}",
                    int(score),
                ])


# ============================================================
# Single-dataset helpers
# ============================================================

def write_corpus_jsonl(corpus_dict, filepath):
    """Write corpus documents to a JSONL file (no ID prefixing)."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        for doc_id, doc in corpus_dict.items():
            entry = {
                "_id": str(doc_id),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
            json.dump(entry, f)
            f.write("\n")


def write_queries_jsonl(queries_dict, filepath):
    """Write queries to a JSONL file (no ID prefixing)."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        for q_id, q_text in queries_dict.items():
            entry = {"_id": str(q_id), "text": q_text}
            json.dump(entry, f)
            f.write("\n")


def write_qrels_tsv(qrels_dict, filepath):
    """Write relevance judgments to a TSV file (no ID prefixing)."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        for q_id, doc_map in qrels_dict.items():
            for doc_id, score in doc_map.items():
                writer.writerow([str(q_id), str(doc_id), int(score)])


# ============================================================
# Text processing helpers
# ============================================================

def model_short_name(full_name):
    """Derive a filesystem-safe short name from a HuggingFace model path.

    Examples:
        'BAAI/bge-m3'                      -> 'bge-m3'
        'sentence-transformers/all-MiniLM'  -> 'all-MiniLM'
    """
    return full_name.split("/")[-1]


def stem_and_tokenize(text, stemmer=None):
    """Lowercase and split, optionally applying stemming per token."""
    tokens = text.lower().split()
    if stemmer is None:
        return tokens
    return [stemmer.stem(w) for w in tokens]


# Module-level stemmer used by each worker process (set via _init_worker).
_worker_stemmer = None


def _init_worker(stemmer_lang, use_stemming=True):
    """ProcessPoolExecutor initializer: construct one SnowballStemmer per
    worker process so it is reused across all batches assigned to that
    worker instead of being recreated on every call.
    """
    from nltk.stem.snowball import SnowballStemmer

    global _worker_stemmer
    if use_stemming:
        _worker_stemmer = SnowballStemmer(stemmer_lang)
    else:
        _worker_stemmer = None


def stem_batch_worker(args):
    """Multiprocessing worker: stem-tokenize a batch of documents.

    Relies on *_worker_stemmer* being initialised by _init_worker so
    the stemmer object is shared across all calls within the same
    worker process.
    Returns a list of JSON-encoded strings ready to be written.
    """
    batch_ids, batch_texts = args
    results = []
    for doc_id, text in zip(batch_ids, batch_texts):
        if _worker_stemmer is None:
            tokens = text.lower().split()
        else:
            tokens = [_worker_stemmer.stem(w) for w in text.lower().split()]
        results.append(json.dumps({"_id": doc_id, "tokens": tokens}))
    return results
