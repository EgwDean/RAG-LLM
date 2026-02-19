import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon

__all__ = [
    'sigmoid',
    'get_query_distribution',
    'compute_divergence_alpha',
    'apply_rrf_fusion',
    'apply_wrrf_fusion',
    'get_sorted_docs'
]

def sigmoid(x, k=10, x0=0):
    """Standard sigmoid, shifted by x0 and scaled by k."""
    return 1 / (1 + np.exp(-k * (x - x0)))



def get_query_distribution(query_tokens, global_counts, total_tokens):
    """Return P(w|Q) and P(w|C) arrays over unique query terms.
    
    Uses add-one smoothing only for terms absent from the corpus,
    so that KL / JS divergence remains well-defined.
    """
    q_len = len(query_tokens)
    if q_len == 0:
        return None, None

    term_freq = Counter(query_tokens)
    unique_tokens = list(term_freq.keys())

    p_corpus = np.empty(len(unique_tokens))
    p_query = np.empty(len(unique_tokens))

    for i, t in enumerate(unique_tokens):
        count = global_counts.get(t, 0)
        if count == 0:
            count = 1
        p_corpus[i] = count / total_tokens
        p_query[i] = term_freq[t] / q_len

    return p_query, p_corpus

def compute_divergence_alpha(query_tokens, freq_data, config, method="jsd", use_sigmoid=False, normalize_01=False, div_stats=None):
    """Compute the dense-weight alpha from query–corpus divergence.

    High divergence means the query is specific — lean towards BM25
    (low alpha). Low divergence means the query is generic — lean
    towards dense retrieval (high alpha).
    
    Args:
        normalize_01: If True, normalize divergence to [0,1] using min/max from div_stats
        div_stats: Dict with 'min' and 'max' keys for normalization
    """
    p_q, p_c = get_query_distribution(
        query_tokens, freq_data['counts'], freq_data['total_tokens']
    )

    if p_q is None:
        return 0.5

    if "jsd" in method:
        div_score = jensenshannon(p_q, p_c, base=2) ** 2
    elif "kld" in method:
        div_score = np.sum(p_q * np.log(p_q / p_c))
    else:
        div_score = 0.5

    # Normalize divergence to [0, 1] if requested
    if normalize_01 and div_stats is not None:
        div_min, div_max = div_stats['min'], div_stats['max']
        if div_max > div_min:
            div_score = (div_score - div_min) / (div_max - div_min)
        else:
            div_score = 0.5

    if use_sigmoid:
        params = config['benchmark']['sigmoid']
        alpha = sigmoid(params['center'] - div_score, k=params['slope'])
    else:
        alpha = 1.0 - div_score

    return float(np.clip(alpha, 0.0, 1.0))

def apply_rrf_fusion(sparse_rankings, dense_rankings, k_rrf):
    """Reciprocal Rank Fusion over two ranked lists."""
    fused = {}
    for rank, doc in enumerate(sparse_rankings):
        fused[doc] = fused.get(doc, 0.0) + 1.0 / (k_rrf + rank + 1)
    for rank, doc in enumerate(dense_rankings):
        fused[doc] = fused.get(doc, 0.0) + 1.0 / (k_rrf + rank + 1)
    return fused

def apply_wrrf_fusion(sparse_rankings, dense_rankings, alpha, k_rrf):
    """Weighted Reciprocal Rank Fusion.

    Like RRF, but each list's contribution is scaled by a weight:
    alpha for dense, (1 - alpha) for sparse.
    """
    fused = {}
    for rank, doc in enumerate(sparse_rankings):
        fused[doc] = fused.get(doc, 0.0) + (1.0 - alpha) / (k_rrf + rank + 1)
    for rank, doc in enumerate(dense_rankings):
        fused[doc] = fused.get(doc, 0.0) + alpha / (k_rrf + rank + 1)
    return fused

def get_sorted_docs(scores_dict):
    """Return doc IDs sorted by descending score."""
    return sorted(scores_dict, key=scores_dict.get, reverse=True)
