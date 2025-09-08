"""
NMF (Non-negative Matrix Factorization) con Keras 3 (backend Torch)
— compatible con tus scripts SVD/ALS/Item–Item (eval rápida @K).

- Embeddings de usuario e ítem con restricción de NO NEGATIVIDAD.
- Predicción = producto punto (sin sesgos, NMF “clásico”).
- Entrena con MSE sobre pares observados (implícitos 0/1 o ratings).
- Evalúa Recall@K y NDCG@K sobre un pool fijo de candidatos.
- Flags de velocidad: --sample_items, --max_eval_users, --skip_save.
- Guarda solo pesos (Keras 3 + Torch no exporta .keras completo).

Uso:
  KERAS_BACKEND=torch python NMF_CF.py ^
    --datadir data --factors 64 --epochs 20 --batch_size 2048 --l2 1e-6 ^
    --k_list 10,20 --sample_items 2000 --max_eval_users 1000 --skip_save
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse, sys, time, json, datetime
import numpy as np
import keras
from keras import layers, regularizers, constraints

# ---------- utilidades ----------
def log(msg): sys.stdout.write(msg + "\n") or sys.stdout.flush()

def load_npz(path):
    d = np.load(path)
    return d["user_idx"], d["item_idx"], d["label"]

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

def check_backend():
    try:
        bk = keras.config.backend()
    except Exception:
        bk = os.environ.get("KERAS_BACKEND", "tensorflow")
    if bk != "torch":
        raise SystemExit(f"Este script espera KERAS_BACKEND=torch, pero encontró '{bk}'.")

# ---------- modelo ----------
def build_nmf(n_users, n_items, factors=64, l2=1e-6):
    user_in = keras.Input(shape=(), dtype="int32", name="user_idx")
    item_in = keras.Input(shape=(), dtype="int32", name="item_idx")

    # Embeddings NO negativos (constraint) + regularización L2
    user_emb = layers.Embedding(
        n_users, factors,
        embeddings_initializer="he_uniform",
        embeddings_regularizer=regularizers.l2(l2),
        embeddings_constraint=constraints.NonNeg(),
        name="user_embedding"
    )(user_in)
    item_emb = layers.Embedding(
        n_items, factors,
        embeddings_initializer="he_uniform",
        embeddings_regularizer=regularizers.l2(l2),
        embeddings_constraint=constraints.NonNeg(),
        name="item_embedding"
    )(item_in)

    # Asegurar salida no negativa también a nivel activación
    user_vec = layers.Flatten()(user_emb)
    user_vec = layers.ReLU()(user_vec)
    item_vec = layers.Flatten()(item_emb)
    item_vec = layers.ReLU()(item_vec)

    pred = layers.Dot(axes=1, name="prediction")([user_vec, item_vec])  # >= 0

    model = keras.Model([user_in, item_in], pred, name="nmf_nonneg_mf")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model

# ---------- evaluación ----------
def evaluate_retrieval(model, te_u, te_i, te_y, n_items,
                       k_list=(10,20), sample_items=2000, max_eval_users=1000, seed=42):
    rng = np.random.default_rng(seed)

    # positivos por usuario en TEST
    test_pos = {}
    for u, i, y in zip(te_u, te_i, te_y):
        if y > 0:
            test_pos.setdefault(int(u), set()).add(int(i))

    eval_users = [u for u, s in test_pos.items() if len(s) > 0]
    if not eval_users:
        return {f"recall@{k}": 0.0 for k in k_list} | {f"ndcg@{k}": 0.0 for k in k_list}

    if max_eval_users and len(eval_users) > max_eval_users:
        eval_users = rng.choice(eval_users, size=max_eval_users, replace=False)

    base_cands = np.unique(rng.choice(n_items, size=min(sample_items, n_items), replace=False))
    fixed_L = None
    rows = []

    t0 = time.time()
    for idx, u in enumerate(eval_users, 1):
        gt = test_pos[int(u)]
        cand = base_cands
        if gt:
            cand = np.unique(np.concatenate([base_cands, np.fromiter(gt, dtype=np.int32)]))

        # fijar tamaño L del candidato para apilar luego
        if fixed_L is None:
            fixed_L = len(cand)
        if len(cand) > fixed_L:
            gt_arr = np.array(list(gt), dtype=np.int32)
            extra  = np.setdiff1d(cand, gt_arr, assume_unique=True)
            n_extra = fixed_L - len(gt_arr)
            if n_extra > 0:
                extra_sample = rng.choice(extra, size=min(n_extra, len(extra)), replace=False)
                cand = np.concatenate([gt_arr, extra_sample]) if len(gt_arr) > 0 else extra_sample
            else:
                cand = gt_arr[:fixed_L]
        elif len(cand) < fixed_L:
            missing = fixed_L - len(cand)
            pool    = np.setdiff1d(np.arange(n_items, dtype=np.int32), cand, assume_unique=True)
            fill    = rng.choice(pool, size=min(missing, len(pool)), replace=False) if missing > 0 else np.array([], dtype=np.int32)
            cand    = np.concatenate([cand, fill]) if len(fill) > 0 else cand

        user_arr = np.full_like(cand, int(u), dtype=np.int32)
        scores = model.predict({"user_idx": user_arr, "item_idx": cand}, verbose=0).reshape(-1)

        order  = np.argsort(-scores)
        ranked = (cand[order][:, None] == np.array(list(gt), dtype=np.int32)).any(axis=1).astype(np.int32)
        rows.append(ranked)

        if idx % 100 == 0:
            log(f"  evaluated users: {idx}/{len(eval_users)}  (elapsed {time.time()-t0:.1f}s)")

    ranked_labels = np.stack(rows, axis=0)
    out = {}
    for k in k_list:
        out[f"recall@{k}"] = recall_at_k(ranked_labels, k)
        out[f"ndcg@{k}"]   = ndcg_at_k(ranked_labels, k)
    return out

# ---------- main ----------
def main():
    check_backend()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--k_list", default="10,20")
    ap.add_argument("--sample_items", type=int, default=2000)
    ap.add_argument("--max_eval_users", type=int, default=1000)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--skip_save", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.random_seed)

    tr_u, tr_i, tr_y = load_npz(os.path.join(args.datadir, "train.npz"))
    va_u, va_i, va_y = load_npz(os.path.join(args.datadir, "valid.npz"))
    te_u, te_i, te_y = load_npz(os.path.join(args.datadir, "test.npz"))

    n_users = int(max(tr_u.max(), va_u.max(), te_u.max())) + 1
    n_items = int(max(tr_i.max(), va_i.max(), te_i.max())) + 1

    model = build_nmf(n_users, n_items, factors=args.factors, l2=args.l2)

    log("▶️  Training…")
    _ = model.fit(
        {"user_idx": tr_u, "item_idx": tr_i},
        tr_y,
        validation_data=({"user_idx": va_u, "item_idx": va_i}, va_y),
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        verbose=1,
    )
    log("✅ Training finished.")

    k_list = tuple(int(x) for x in args.k_list.split(","))
    log(f"🧪 Evaluating: sample_items={args.sample_items}, max_eval_users={args.max_eval_users}, Ks={k_list}")
    metrics = evaluate_retrieval(model, te_u, te_i, te_y, n_items,
                                 k_list=k_list,
                                 sample_items=args.sample_items,
                                 max_eval_users=args.max_eval_users)
    log(f"📊 Test metrics (subset): {metrics}")

    # Guardado (pesos + métricas)
    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Results'))
    script_name  = os.path.splitext(os.path.basename(__file__))[0]
    save_dir     = os.path.join(results_root, script_name)
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_path = os.path.join(save_dir, f"{script_name}.{ts}.weights.h5")
    metrics_path = os.path.join(save_dir, f"{script_name}.{ts}.metrics.json")

    if not args.skip_save:
        log("💾 Guardando pesos y métricas…")
        model.save_weights(weights_path)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log(f"✅ Guardado en {save_dir}")
    else:
        log("⏭️  Skipped saving.")

if __name__ == "__main__":
    main()
