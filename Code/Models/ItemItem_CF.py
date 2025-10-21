import numpy as np
import scipy.sparse as sp

# ===== Helpers de similitud =====
def _row_mean_center(csr: sp.csr_matrix) -> sp.csr_matrix:
    X = csr.tocsr(copy=True)
    indptr, data = X.indptr, X.data
    for r in range(X.shape[0]):
        start, end = indptr[r], indptr[r+1]
        if end > start:
            m = data[start:end].mean()
            data[start:end] -= m
    X.eliminate_zeros()
    return X

def item_item_similarity(ui: sp.csr_matrix, K: int, similarity: str = "cosine") -> sp.csr_matrix:
    X = ui.tocsr()
    if similarity == "adjusted_cosine":
        X = _row_mean_center(X)

    XtX = (X.T @ X).tocsr().astype(np.float32)
    norms = np.sqrt(X.multiply(X).sum(axis=0)).A1.astype(np.float32) + 1e-8
    inv = 1.0 / norms
    D_inv = sp.diags(inv, dtype=np.float32)
    S = (D_inv @ XtX @ D_inv).tocsr()
    S.setdiag(0.0); S.eliminate_zeros()

    S = S.tolil()
    I = S.shape[0]
    for r in range(I):
        row = S.rows[r]
        if not row:
            continue
        data = np.asarray(S.data[r], dtype=np.float32)
        if len(row) > K:
            idx = np.argpartition(data, -K)[-K:]
            keep = set(np.array(row)[idx].tolist())
            new_rows, new_data = [], []
            for c, v in zip(row, data):
                if c in keep:
                    new_rows.append(c); new_data.append(float(v))
            S.rows[r] = new_rows; S.data[r] = new_data
    S = S.tocsr(); S.eliminate_zeros()
    return S


# ===== Adaptador =====
class Recommender:
    def __init__(self, K=150, similarity="cosine"):
        self.K = K
        self.similarity = similarity  # "cosine" o "adjusted_cosine"
        self.S = None
        self.ui_train = None
        self.n_items = None

    def fit(self, train_csr):
        n_users, n_items = train_csr.shape
        self.n_items = n_items
        u, i = train_csr.nonzero()
        y = np.ones_like(u, dtype=np.float32)
        self.ui_train = sp.csr_matrix((y, (u, i)), shape=(n_users, n_items), dtype=np.float32)
        self.S = item_item_similarity(self.ui_train, K=self.K, similarity=self.similarity)

    def recommend(self, user_id, k=10, exclude_seen=None):
        seen = self.ui_train[int(user_id)].indices
        scores = np.zeros(self.n_items, dtype=np.float32)
        if len(seen) > 0:
            scores = np.asarray(self.S[seen].sum(axis=0)).ravel().astype(np.float32)
        if exclude_seen is not None and len(seen) > 0:
            scores[seen] = -1e9
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, k)[:k]
            order = idx[np.argsort(-scores[idx])]
        return order.tolist()
