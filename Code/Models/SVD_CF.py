"""
SVD-style (biased MF) with Keras 3 on Torch backend — FAST eval.

Improvements vs previous version:
- Progress prints (train end, eval start, saving...).
- CLI flags: --max_eval_users (limit #users to evaluate), --skip_save (avoid model.save).
- Faster eval by limiting candidates (sample_items) and users.

Usage:
  KERAS_BACKEND=torch python svd_collaborative_filtering.py \
    --datadir data --factors 64 --epochs 20 --batch_size 2048 --l2 1e-6 \
    --k_list 10,20 --sample_items 2000 --max_eval_users 1000 --skip_save
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse, os, numpy as np, json, sys, time

import keras  # Keras 3 with Torch backend
from keras import layers, regularizers

def check_backend():
    try:
        bk = keras.config.backend()
    except Exception:
        import os
        bk = os.environ.get("KERAS_BACKEND", "tensorflow")
    if bk != "torch":
        raise SystemExit(f"This script expects KERAS_BACKEND=torch, but found '{bk}'. Set env var KERAS_BACKEND=torch and retry.")

def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def load_npz(path):
    d = np.load(path)
    return d["user_idx"], d["item_idx"], d["label"]

def build_biased_mf(n_users, n_items, factors=64, l2=1e-6):
    user_in = keras.Input(shape=(), dtype="int32", name="user_idx")
    item_in = keras.Input(shape=(), dtype="int32", name="item_idx")

    user_emb = layers.Embedding(n_users, factors,
                                embeddings_regularizer=regularizers.l2(l2),
                                name="user_embedding")(user_in)
    item_emb = layers.Embedding(n_items, factors,
                                embeddings_regularizer=regularizers.l2(l2),
                                name="item_embedding")(item_in)

    user_vec = layers.Flatten()(user_emb)
    item_vec = layers.Flatten()(item_emb)

    user_bias = layers.Embedding(n_users, 1, name="user_bias")(user_in)
    item_bias = layers.Embedding(n_items, 1, name="item_bias")(item_in)
    user_bias = layers.Flatten()(user_bias)
    item_bias = layers.Flatten()(item_bias)

    dot = layers.Dot(axes=1)([user_vec, item_vec])
    pred = layers.Add(name="prediction")([dot, user_bias, item_bias])

    model = keras.Model(inputs=[user_in, item_in], outputs=pred, name="svd_biased_mf_torch_fast")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model

def evaluate_retrieval(model, test_u, test_i, test_y, n_items, k_list=(10,20),
                        sample_items=2000, max_eval_users=1000, seed=42):
    rng = np.random.default_rng(seed)

    # ground-truth positives per user
    test_pos = {}
    for u, i, y in zip(test_u, test_i, test_y):
        if y > 0:
            test_pos.setdefault(int(u), set()).add(int(i))

    eval_users = [u for u, s in test_pos.items() if len(s) > 0]
    if not eval_users:
        return {f"recall@{k}": 0.0 for k in k_list} | {f"ndcg@{k}": 0.0 for k in k_list}

    # limit users for speed
    if max_eval_users and len(eval_users) > max_eval_users:
        eval_users = rng.choice(eval_users, size=max_eval_users, replace=False)

    candidates = np.unique(rng.choice(n_items, size=min(sample_items, n_items), replace=False))

    from utils_metrics import recall_at_k, ndcg_at_k
    ranked_labels = []

    t0 = time.time()
    # Para asegurar que todos los arrays tengan la misma longitud:
    fixed_cand_size = None
    for idx, u in enumerate(eval_users, 1):
        gt = test_pos[u]
        cand = candidates
        if gt:
            cand = np.unique(np.concatenate([candidates, np.fromiter(gt, dtype=np.int32)]))
        # Fijar tamaño de candidatos en la primera iteración
        if fixed_cand_size is None:
            fixed_cand_size = len(cand)
        # Si el tamaño cambia, recortar o rellenar
        if len(cand) > fixed_cand_size:
            # recortar aleatoriamente (pero siempre incluir los ground-truth)
            gt_arr = np.array(list(gt), dtype=np.int32)
            # asegurarse de que gt esté incluido
            extra = np.setdiff1d(cand, gt_arr, assume_unique=True)
            n_extra = fixed_cand_size - len(gt_arr)
            if n_extra > 0:
                rng = np.random.default_rng(seed + idx)
                extra_sample = rng.choice(extra, size=n_extra, replace=False) if len(extra) > n_extra else extra
                cand = np.concatenate([gt_arr, extra_sample])
            else:
                cand = gt_arr[:fixed_cand_size]
        elif len(cand) < fixed_cand_size:
            # rellenar con candidatos aleatorios que no estén en cand
            missing = fixed_cand_size - len(cand)
            pool = np.setdiff1d(np.arange(n_items), cand, assume_unique=True)
            rng = np.random.default_rng(seed + idx)
            fill = rng.choice(pool, size=missing, replace=False) if len(pool) >= missing else pool
            cand = np.concatenate([cand, fill])
        # cand ahora tiene tamaño fijo
        user_arr = np.full_like(cand, u, dtype=np.int32)
        scores = model.predict({"user_idx": user_arr, "item_idx": cand}, verbose=0).reshape(-1)

        order = np.argsort(-scores)
        ranked = (cand[order][:, None] == np.array(list(gt), dtype=np.int32)).any(axis=1).astype(np.int32)
        ranked_labels.append(ranked)

        if idx % 100 == 0:
            log(f"  evaluated users: {idx}/{len(eval_users)}  (elapsed {time.time()-t0:.1f}s)")

    ranked_labels = np.stack(ranked_labels, axis=0)
    metrics = {}
    for k in k_list:
        metrics[f"recall@{k}"] = recall_at_k(ranked_labels, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_labels, k)
    return metrics

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
    ap.add_argument("--skip_save", action="store_true")
    args = ap.parse_args()

    tr_u, tr_i, tr_y = load_npz(os.path.join(args.datadir, "train.npz"))
    va_u, va_i, va_y = load_npz(os.path.join(args.datadir, "valid.npz"))
    te_u, te_i, te_y = load_npz(os.path.join(args.datadir, "test.npz"))

    n_users = int(max(tr_u.max(), va_u.max(), te_u.max())) + 1
    n_items = int(max(tr_i.max(), va_i.max(), te_i.max())) + 1

    model = build_biased_mf(n_users, n_items, factors=args.factors, l2=args.l2)

    log("▶️  Training...")
    history = model.fit(
        {"user_idx": tr_u, "item_idx": tr_i},
        tr_y,
        validation_data=({"user_idx": va_u, "item_idx": va_i}, va_y),
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
    )
    log("✅ Training finished.")

    # Evaluate
    k_list = tuple(int(x) for x in args.k_list.split(","))
    log(f"🧪 Evaluating: sample_items={args.sample_items}, max_eval_users={args.max_eval_users}, Ks={k_list}")
    metrics = evaluate_retrieval(model, te_u, te_i, te_y, n_items,
                                 k_list=k_list,
                                 sample_items=args.sample_items,
                                 max_eval_users=args.max_eval_users)
    log(f"📊 Test metrics (subset): {metrics}")

    # Save
    import datetime
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    # Save to Results/<script_name>/
    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Results'))
    save_dir = os.path.join(results_root, script_name)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_path = os.path.join(save_dir, f"{script_name}.{timestamp}.weights.h5")
    metrics_path = os.path.join(save_dir, f"{script_name}.{timestamp}.metrics.json")

    if not args.skip_save:
        log("⚠️  Model export/save is not supported with Keras 3 + Torch backend. Saving weights only.")
        model.save_weights(weights_path)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log(f"✅ Model weights and metrics saved to {save_dir}.")
    else:
        log("⏭️  Skipped saving.")

if __name__ == "__main__":
    main()
