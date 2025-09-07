
"""
Utility metrics for retrieval evaluation: Recall@K and NDCG@K.
"""
import numpy as np

def recall_at_k(ranked_labels, k=10):
    ranked = np.asarray(ranked_labels)
    topk = ranked[:, :k]
    denom = np.maximum(ranked.sum(axis=1), 1e-8)
    return float((topk.sum(axis=1) / denom).mean())

def dcg_at_k(rel, k=10):
    rel = np.asarray(rel)[:, :k]
    gains = (2.0**rel - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, rel.shape[1] + 2))
    return (gains * discounts).sum(axis=1)

def ndcg_at_k(ranked_labels, k=10):
    ranked = np.asarray(ranked_labels)
    ideal = np.sort(ranked, axis=1)[:, ::-1]
    dcg = dcg_at_k(ranked, k)
    idcg = dcg_at_k(ideal, k) + 1e-8
    return float((dcg / idcg).mean())
