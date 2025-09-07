
"""
Data prep script (auto-detect columns):
Parquet -> dense id mapping -> temporal split -> NPZ shards.

USAGE (mínimo):
  python data_prep_to_npz.py --parquet Data/dataset_procesado.parquet --outdir data

Opcionales (para forzar nombres si querés):
  --time_col ...  --label_col ...  --user_col ...  --item_col ...
  --valid_ratio 0.1 --test_ratio 0.1
"""
import argparse
import os
import re
import pandas as pd
import numpy as np

ALIASES = {
    "user": ["user_id","userid","user","steamid","profile_id","u_id","id_user","author_id"],
    "item": ["game_id","item_id","item","app_id","appid","game","id_game","product_id","content_id"],
    "time": ["timestamp","time","datetime","created_at","review_time","event_time","ts","date"],
    "label": ["rating","score","label","recommended","like","target","y"]
}

def pick_column(df, provided, kind):
    if provided and provided in df.columns:
        return provided
    # 1) exact aliases
    for cand in ALIASES[kind]:
        if cand in df.columns:
            return cand
    # 2) regex heuristics
    patterns = {
        "user": re.compile(r"user|steam|profile", re.I),
        "item": re.compile(r"game|item|app|product|content", re.I),
        "time": re.compile(r"time|date|ts", re.I),
        "label": re.compile(r"rating|score|label|reco|target|like|y", re.I),
    }
    rx = patterns[kind]
    for c in df.columns:
        if rx.search(str(c)):
            return c
    return None

def make_dense_codes(series):
    cat = series.astype("category")
    codes = cat.cat.codes.astype("int32")
    # mapping table
    mp = pd.DataFrame({"raw": cat.astype(str).values, "idx": codes.astype("int64").values})
    mp = mp.drop_duplicates("idx").sort_values("idx").reset_index(drop=True)
    return codes, mp

def temporal_split(df, time_col, valid_ratio=0.1, test_ratio=0.1):
    if time_col not in df.columns:
        # if no time column, create a synthetic order to allow deterministic split
        df = df.copy()
        df["_synthetic_time"] = np.arange(len(df), dtype=np.int64)
        time_col = "_synthetic_time"
    ts = df[time_col]
    q_train = 1.0 - (valid_ratio + test_ratio)
    cutoff1 = ts.quantile(q_train)
    cutoff2 = ts.quantile(q_train + valid_ratio)
    train = df[ts <= cutoff1]
    valid = df[(ts > cutoff1) & (ts <= cutoff2)]
    test  = df[ts > cutoff2]
    return train, valid, test, time_col, cutoff1, cutoff2

def coerce_label(series):
    # Convert to float32; handle bools/strings ('True'/'False', 'yes'/'no')
    s = series.copy()
    if s.dtype == bool:
        return s.astype("float32")
    # try to map common strings
    lower = s.astype(str).str.lower()
    mask_bool = lower.isin(["true","false","yes","no","y","n","1","0"])
    if mask_bool.all():
        mapped = lower.map({"true":1.0,"false":0.0,"yes":1.0,"no":0.0,"y":1.0,"n":0.0,"1":1.0,"0":0.0})
        return mapped.astype("float32")
    # fallback: numeric cast
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype("float32")

def to_npz_split(split_df, out_path, user_col, item_col, label_col):
    users = split_df[user_col].to_numpy(dtype=np.int32)
    items = split_df[item_col].to_numpy(dtype=np.int32)
    if label_col and label_col in split_df.columns:
        labels = coerce_label(split_df[label_col]).to_numpy(dtype="float32")
    else:
        labels = np.ones(len(split_df), dtype=np.float32)
    np.savez_compressed(out_path, user_idx=users, item_idx=items, label=labels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--time_col", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--user_col", default=None)
    ap.add_argument("--item_col", default=None)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Read parquet (needs pyarrow or fastparquet installed in your env)
    df = pd.read_parquet(args.parquet, engine="pyarrow")

    # Auto-pick columns
    user_raw = pick_column(df, args.user_col, "user")
    item_raw = pick_column(df, args.item_col, "item")
    time_col = pick_column(df, args.time_col, "time") if args.time_col is None else args.time_col
    label_col = pick_column(df, args.label_col, "label") if args.label_col is None else args.label_col

    if user_raw is None or item_raw is None:
        raise SystemExit(f"No pude detectar columnas de usuario/item. Columnas disponibles: {df.columns.tolist()}")

    # Build dense ids
    df = df.copy()
    df["user_idx"], user_map = make_dense_codes(df[user_raw].astype(str))
    df["item_idx"], item_map = make_dense_codes(df[item_raw].astype(str))

    # Sort by time if present
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)

    # Temporal split
    train, valid, test, used_time_col, cut1, cut2 = temporal_split(df, time_col if time_col else "_synthetic_time",
                                                                   args.valid_ratio, args.test_ratio)

    # Save mappings and stats
    user_map.to_csv(os.path.join(args.outdir, "user_map.csv"), index=False)
    item_map.to_csv(os.path.join(args.outdir, "item_map.csv"), index=False)

    stats = {
        "n_rows": int(len(df)),
        "n_users": int(df["user_idx"].nunique()),
        "n_items": int(df["item_idx"].nunique()),
        "used_columns": {"user_raw": user_raw, "item_raw": item_raw, "time_col": used_time_col, "label_col": label_col},
        "split_sizes": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))}
    }
    with open(os.path.join(args.outdir, "split_stats.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Save NPZ shards
    to_npz_split(train, os.path.join(args.outdir, "train.npz"), "user_idx", "item_idx", label_col)
    to_npz_split(valid, os.path.join(args.outdir, "valid.npz"), "user_idx", "item_idx", label_col)
    to_npz_split(test,  os.path.join(args.outdir, "test.npz"),  "user_idx", "item_idx", label_col)

    print("Done. Wrote:", args.outdir)
    print("Stats:", stats)

if __name__ == "__main__":
    main()
