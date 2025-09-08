"""
ALS (Implicit MF) — evaluación rápida y compatible con tu SVD.

- Entrena ALS (Hu–Koren–Volinsky) sobre feedback implícito binario (0/1).
- Pondera la matriz con 'alpha' (confianza).
- Evalúa Recall@K y NDCG@K sobre un pool de candidatos con tamaño fijo.
- Flags de velocidad: --sample_items, --max_eval_users, --skip_save.
- Guarda factores y métricas en Results/<script_name>/ via timestamp.

Uso:
  python als_collaborative_filtering.py ^
    --datadir data --factors 64 --iterations 20 --reg 0.01 --alpha 40 ^
    --k_list 10,20 --sample_items 2000 --max_eval_users 1000 --skip_save
"""

import argparse, os, sys, time, json, datetime
import numpy as np
import scipy.sparse as sp

def log(msg: str):
    sys.stdout.write(msg + "\n"); sys.stdout.flush()

def load_npz(path):
    d = np.load(path)
    return d["user_idx"], d["item_idx"], d["label"]

def build_ui_matrix(n_users, n_items, u, i, y, alpha: float):
    # Implícitos: confianza = 1 + alpha * r; para ALS de 'implicit' se usa data = alpha * r
    data = (y.astype(np.float32) * alpha)
    ui = sp.csr_matrix((data, (u, i)), shape=(n_users, n_items), dtype=np.float32)
    return ui

def recall_at_k(ranked_labels: np.ndarray, k: int) -> float:
    # ranked_labels: [num_users, L] binario con 1 si el item en esa posición es relevante
    topk = ranked_labels[:, :k]
    # cada fila puede tener >> 1 relevantes (si tu test lo permite)
    denom = np.maximum(1, ranked_labels.sum(axis=1))
    return float((topk.sum(axis=1) / denom).mean())

def ndcg_at_k(ranked_labels: np.ndarray, k: int) -> float:
    k = min(k, ranked_labels.shape[1])
    gains = 1.0 / np.log2(2 + np.arange(k, dtype=np.float32))
    dcg = (ranked_labels[:, :k] * gains).sum(axis=1)
    # IDCG: top 'r' relevantes en las primeras posiciones
    r = ranked_labels.sum(axis=1).clip(min=1)
    idcg = np.array([gains[:int(min(k, rr))].sum() for rr in r], dtype=np.float32)
    return float((dcg / idcg).mean())

def evaluate(model, n_items, te_u, te_i, te_y, tr_u, k_list=(10,20),
             sample_items=2000, max_eval_users=1000, seed=42):
    rng = np.random.default_rng(seed)

    # Ground-truth positivos por usuario
    test_pos = {}
    for u, i, y in zip(te_u, te_i, te_y):
        if y > 0:
            test_pos.setdefault(int(u), set()).add(int(i))

    eval_users = [u for u, s in test_pos.items() if len(s) > 0]
    # Filtrar usuarios que no están en train
    train_users = set(int(x) for x in np.unique(tr_u))
    U = model.user_factors   # [n_users, f]
    n_users_model = U.shape[0]
    eval_users = [u for u in eval_users if u in train_users and u < n_users_model]
    if max_eval_users and len(eval_users) > max_eval_users:
        eval_users = rng.choice(eval_users, size=max_eval_users, replace=False)

    if len(eval_users) == 0:
        return {f"recall@{k}": 0.0 for k in k_list} | {f"ndcg@{k}": 0.0 for k in k_list}

    # Pool global de candidatos (tamaño base)
    base_cands = np.unique(rng.choice(n_items, size=min(sample_items, n_items), replace=False))
    fixed_cand_size = None

    ranked_rows = []
    t0 = time.time()

    U = model.user_factors   # [n_users, f]
    V = model.item_factors   # [n_items, f]

    for idx, u in enumerate(eval_users, 1):
        gt = test_pos[int(u)]
        # incluir todos los positivos del usuario
        cand = base_cands
        if gt:
            cand = np.unique(np.concatenate([base_cands, np.fromiter(gt, dtype=np.int32)]))

        # fijar tamaño y normalizar tamaño del candidato
        if fixed_cand_size is None:
            fixed_cand_size = len(cand)
        if len(cand) > fixed_cand_size:
            gt_arr = np.array(list(gt), dtype=np.int32)
            extra = np.setdiff1d(cand, gt_arr, assume_unique=True)
            n_extra = fixed_cand_size - len(gt_arr)
            if n_extra > 0:
                extra_sample = rng.choice(extra, size=min(n_extra, len(extra)), replace=False)
                cand = np.concatenate([gt_arr, extra_sample]) if len(gt_arr) > 0 else extra_sample
            else:
                cand = gt_arr[:fixed_cand_size]
        elif len(cand) < fixed_cand_size:
            missing = fixed_cand_size - len(cand)
            pool = np.setdiff1d(np.arange(n_items, dtype=np.int32), cand, assume_unique=True)
            fill = rng.choice(pool, size=min(missing, len(pool)), replace=False) if missing > 0 else np.array([], dtype=np.int32)
            cand = np.concatenate([cand, fill]) if len(fill) > 0 else cand

        # Scoring por producto punto (no filtramos vistos de train para mantener paridad con tu SVD actual)
        u_vec = U[int(u)]
        scores = V[cand] @ u_vec
        order = np.argsort(-scores)
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
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--reg", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=40.0)
    ap.add_argument("--k_list", default="10,20")
    ap.add_argument("--sample_items", type=int, default=2000)
    ap.add_argument("--max_eval_users", type=int, default=1000)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--skip_save", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.random_seed)

    # Cargar splits
    tr_u, tr_i, tr_y = load_npz(os.path.join(args.datadir, "train.npz"))
    va_u, va_i, va_y = load_npz(os.path.join(args.datadir, "valid.npz"))
    te_u, te_i, te_y = load_npz(os.path.join(args.datadir, "test.npz"))

    n_users = int(max(tr_u.max(), va_u.max(), te_u.max())) + 1
    n_items = int(max(tr_i.max(), va_i.max(), te_i.max())) + 1

    # Matriz usuario–ítem con ponderación alpha (implícitos)
    log("🔧 Construyendo matriz UI…")
    ui = build_ui_matrix(n_users, n_items, tr_u, tr_i, tr_y, args.alpha)  # csr [U, I]

    # 'implicit' entrena con matriz item–user (traspuesta)
    iu = ui.T.tocsr()

    # Entrenar ALS
    log(f"▶️  Entrenando ALS: factors={args.factors}, iterations={args.iterations}, reg={args.reg}, alpha={args.alpha}")
    try:
        import implicit
        from implicit.als import AlternatingLeastSquares
    except Exception as e:
        raise SystemExit("Falta instalar 'implicit'. Ejecutá: pip install implicit") from e

    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.reg,
        iterations=args.iterations,
        random_state=args.random_seed,
        use_gpu=False  # poné True si tenés CUDA y querés probar
    )
    # Nota: AlternatingLeastSquares espera item-user (iu)
    model.fit(iu)
    log("✅ Entrenamiento terminado.")

    # Evaluación
    k_list = tuple(int(x) for x in args.k_list.split(","))
    log(f"🧪 Evaluando: sample_items={args.sample_items}, max_eval_users={args.max_eval_users}, Ks={k_list}")
    metrics = evaluate(model, n_items, te_u, te_i, te_y, tr_u,
                       k_list=k_list,
                       sample_items=args.sample_items,
                       max_eval_users=args.max_eval_users,
                       seed=args.random_seed)
    log(f"📊 Test metrics (subset): {metrics}")

    # Guardado
    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Results'))
    script_name  = os.path.splitext(os.path.basename(__file__))[0]
    save_dir     = os.path.join(results_root, script_name)
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    weights_path = os.path.join(save_dir, f"{script_name}.{ts}.npz")
    metrics_path = os.path.join(save_dir, f"{script_name}.{ts}.metrics.json")

    if not args.skip_save:
        log("💾 Guardando factores (user_factors, item_factors) y métricas…")
        np.savez_compressed(
            weights_path,
            user_factors=model.user_factors.astype(np.float32),
            item_factors=model.item_factors.astype(np.float32),
            factors=np.int32(args.factors),
            iterations=np.int32(args.iterations),
            reg=np.float32(args.reg),
            alpha=np.float32(args.alpha),
            random_seed=np.int32(args.random_seed)
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log(f"✅ Guardado en {save_dir}")
    else:
        log("⏭️  Skipped saving.")

if __name__ == "__main__":
    main()
