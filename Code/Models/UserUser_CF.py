"""
User–User kNN (cosine) — evaluación compatible con tus SVD / ALS / Item–Item.

- Construye matriz usuario–ítem (binaria) con ponderación opcional TF–IDF por ítem.
- Calcula similitud usuario–usuario por coseno y conserva Top-K vecinos por usuario.
- Puntúa con suma ponderada de los vecinos sobre los ítems candidato.
- Evalúa Recall@K y NDCG@K con pool de candidatos de tamaño fijo (incluye los positivos).
- Flags de velocidad: --sample_items, --max_eval_users, --skip_save.
- Guarda la matriz de similitud (sparse) y métricas en Results/<script>/ con timestamp.

Uso:
  python UserUser_CF.py ^
    --datadir data --K 150 --weighting tfidf --k_list 10,20 ^
    --sample_items 2000 --max_eval_users 1000 --skip_save
"""

import argparse, os, sys, time, json, datetime
import numpy as np
import scipy.sparse as sp

def log(msg: str):
    sys.stdout.write(msg + "\n"); sys.stdout.flush()

def load_npz(path):
    d = np.load(path)
    return d["user_idx"], d["item_idx"], d["label"]

def build_ui_matrix(n_users, n_items, u, i, y, weighting: str = "tfidf"):
    """
    Devuelve CSR [U, I] binaria con ponderación opcional:
      - 'none' : sólo binaria (1 si hay interacción)
      - 'tfidf': multiplica columnas (ítems) por IDF para bajar popularidad
    """
    data = (y > 0).astype(np.float32)
    ui = sp.csr_matrix((data, (u, i)), shape=(n_users, n_items), dtype=np.float32)
    ui.data[:] = 1.0
    if weighting == "tfidf":
        df = np.asarray((ui > 0).sum(axis=0)).ravel().astype(np.float32)  # usuarios por ítem
        idf = np.log((n_users + 1.0) / (df + 1.0)) + 1.0
        D = sp.diags(idf, dtype=np.float32)
        ui = ui @ D
    elif weighting == "none":
        pass
    else:
        raise ValueError("weighting debe ser 'none' o 'tfidf'")
    return ui

def cosine_topk_user_user(ui: sp.csr_matrix, K: int) -> sp.csr_matrix:
    """
    S ≈ coseno por filas (usuarios): S = D^{-1} (X X^T) D^{-1}, guardando Top-K por fila y diag=0.
    ui: CSR [U, I]  -> devuelve CSR [U, U] (similitud entre usuarios)
    """
    X = ui.tocsr()
    XXt = (X @ X.T).tocsr().astype(np.float32)  # [U, U]
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1.astype(np.float32) + 1e-8  # ||fila u||
    inv = 1.0 / norms
    D_inv = sp.diags(inv, dtype=np.float32)
    S = (D_inv @ XXt @ D_inv).tocsr()

    # eliminar diagonal
    S.setdiag(0.0); S.eliminate_zeros()

    # Top-K por fila
    S = S.tolil()
    U = S.shape[0]
    for r in range(U):
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
            S.rows[r] = new_rows
            S.data[r] = new_data
    S = S.tocsr(); S.eliminate_zeros()
    return S

def recall_at_k(ranked_labels: np.ndarray, k: int) -> float:
    topk = ranked_labels[:, :k]
    denom = np.maximum(1, ranked_labels.sum(axis=1))
    return float((topk.sum(axis=1) / denom).mean())

def ndcg_at_k(ranked_labels: np.ndarray, k: int) -> float:
    k = min(k, ranked_labels.shape[1])
    gains = 1.0 / np.log2(2 + np.arange(k, dtype=np.float32))
    dcg  = (ranked_labels[:, :k] * gains).sum(axis=1)
    r    = ranked_labels.sum(axis=1).clip(min=1)
    idcg = np.array([gains[:int(min(k, rr))].sum() for rr in r], dtype=np.float32)
    return float((dcg / idcg).mean())

def evaluate_userknn(S: sp.csr_matrix, n_items: int,
                     tr_u, tr_i, te_u, te_i, te_y,
                     k_list=(10,20), sample_items=2000, max_eval_users=1000,
                     seed=42, filter_seen: bool=False):
    rng = np.random.default_rng(seed)

    # Historial de train por usuario (para filtrar vistos si se pide)
    n_users = int(max(tr_u.max(), te_u.max())) + 1
    ui_train = sp.csr_matrix(
        (np.ones_like(tr_i, dtype=np.float32), (tr_u, tr_i)),
        shape=(n_users, n_items), dtype=np.float32
    )
    seen_items_by_u = [ui_train[u].indices for u in range(n_users)]

    # Positivos en test
    test_pos = {}
    for u, i, y in zip(te_u, te_i, te_y):
        if y > 0:
            test_pos.setdefault(int(u), set()).add(int(i))

    eval_users = [u for u, s in test_pos.items() if len(s) > 0]
    if max_eval_users and len(eval_users) > max_eval_users:
        eval_users = rng.choice(eval_users, size=max_eval_users, replace=False)
    if len(eval_users) == 0:
        return {f"recall@{k}": 0.0 for k in k_list} | {f"ndcg@{k}": 0.0 for k in k_list}

    # Pool base de candidatos
    base_cands = np.unique(rng.choice(n_items, size=min(sample_items, n_items), replace=False))
    fixed_cand_size = None

    ranked_rows = []
    t0 = time.time()

    S = S.tocsr()
    for idx, u in enumerate(eval_users, 1):
        gt = test_pos[int(u)]
        cand = base_cands
        if gt:
            cand = np.unique(np.concatenate([base_cands, np.fromiter(gt, dtype=np.int32)]))

        # fijar tamaño de candidatos (para poder apilar)
        if fixed_cand_size is None:
            fixed_cand_size = len(cand)
        if len(cand) > fixed_cand_size:
            gt_arr = np.array(list(gt), dtype=np.int32)
            extra  = np.setdiff1d(cand, gt_arr, assume_unique=True)
            n_extra = fixed_cand_size - len(gt_arr)
            if n_extra > 0:
                extra_sample = rng.choice(extra, size=min(n_extra, len(extra)), replace=False)
                cand = np.concatenate([gt_arr, extra_sample]) if len(gt_arr) > 0 else extra_sample
            else:
                cand = gt_arr[:fixed_cand_size]
        elif len(cand) < fixed_cand_size:
            missing = fixed_cand_size - len(cand)
            pool    = np.setdiff1d(np.arange(n_items, dtype=np.int32), cand, assume_unique=True)
            fill    = rng.choice(pool, size=min(missing, len(pool)), replace=False) if missing > 0 else np.array([], dtype=np.int32)
            cand    = np.concatenate([cand, fill]) if len(fill) > 0 else cand

        # vecinos del usuario u
        row = S.getrow(int(u))
        neigh_idx = row.indices
        neigh_w   = row.data.astype(np.float32)

        if len(neigh_idx) == 0:
            scores = np.zeros_like(cand, dtype=np.float32)
        else:
            # scores(u, cand) = sum_{v in Vecinos(u)} S[u,v] * UI_train[v, cand]
            # UI_train[neigh_idx][:, cand] -> (n_vecinos, L)
            M = ui_train[neigh_idx][:, cand]  # CSR
            scores = (neigh_w @ M.toarray()).astype(np.float32)  # [L]

        if filter_seen:
            seen = set(seen_items_by_u[int(u)])
            mask_seen = np.isin(cand, list(seen), assume_unique=False)
            scores[mask_seen] = -1e9  # mandar vistos al fondo

        order  = np.argsort(-scores)
        ranked = (cand[order][:, None] == np.array(list(gt), dtype=np.int32)).any(axis=1).astype(np.int32)
        ranked_rows.append(ranked)

        if idx % 100 == 0:
            log(f"  evaluated users: {idx}/{len(eval_users)} (elapsed {time.time()-t0:.1f}s)")

    ranked_labels = np.stack(ranked_rows, axis=0)
    out = {}
    for k in k_list:
        out[f"recall@{k}"] = recall_at_k(ranked_labels, k)
        out[f"ndcg@{k}"]   = ndcg_at_k(ranked_labels, k)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--K", type=int, default=150, help="vecinos por usuario")
    ap.add_argument("--weighting", choices=["none","tfidf"], default="tfidf",
                    help="ponderación por ítem antes del coseno (tfidf recomendado)")
    ap.add_argument("--k_list", default="10,20")
    ap.add_argument("--sample_items", type=int, default=2000)
    ap.add_argument("--max_eval_users", type=int, default=1000)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--filter_seen", action="store_true",
                    help="si se pasa, no recomienda ítems vistos en train")
    ap.add_argument("--skip_save", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.random_seed)

    # cargar splits
    tr_u, tr_i, tr_y = load_npz(os.path.join(args.datadir, "train.npz"))
    va_u, va_i, va_y = load_npz(os.path.join(args.datadir, "valid.npz"))
    te_u, te_i, te_y = load_npz(os.path.join(args.datadir, "test.npz"))

    n_users = int(max(tr_u.max(), va_u.max(), te_u.max())) + 1
    n_items = int(max(tr_i.max(), va_i.max(), te_i.max())) + 1

    # matriz usuario–ítem y similitud user–user
    log(f"🔧 Construyendo UI (weighting={args.weighting})…")
    ui = build_ui_matrix(n_users, n_items, tr_u, tr_i, tr_y, weighting=args.weighting)
    log(f"🧮 Calculando similitud user–user (cosine, TopK={args.K})…")
    S = cosine_topk_user_user(ui, K=args.K)  # CSR [U, U]

    # evaluación
    k_list = tuple(int(x) for x in args.k_list.split(","))
    log(f"🧪 Evaluando: sample_items={args.sample_items}, max_eval_users={args.max_eval_users}, Ks={k_list}")
    metrics = evaluate_userknn(S, n_items, tr_u, tr_i, te_u, te_i, te_y,
                               k_list=k_list,
                               sample_items=args.sample_items,
                               max_eval_users=args.max_eval_users,
                               seed=args.random_seed,
                               filter_seen=args.filter_seen)
    log(f"📊 Test metrics (subset): {metrics}")

    # guardado
    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Results'))
    script_name  = os.path.splitext(os.path.basename(__file__))[0]
    save_dir     = os.path.join(results_root, script_name)
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    sim_path     = os.path.join(save_dir, f"{script_name}.{ts}.similarity_topk.npz")
    metrics_path = os.path.join(save_dir, f"{script_name}.{ts}.metrics.json")

    if not args.skip_save:
        log("💾 Guardando matriz de similitud (sparse) y métricas…")
        sp.save_npz(sim_path, S)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log(f"✅ Guardado en {save_dir}")
    else:
        log("⏭️  Skipped saving.")

if __name__ == "__main__":
    main()
