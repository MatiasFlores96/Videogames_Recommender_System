#!/usr/bin/env python3
"""
Inspector de splits .npz (train/test/valid)

Ejemplos de uso:
---------------
python Code/inspect.py --splits_dir Data
python Code/inspect.py --paths Data/train.npz Data/test.npz
"""

import argparse
import json
from pathlib import Path
import numpy as np


def _load_triplet_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.keys())

    # formato 1: rows, cols, data, shape
    if {"rows", "cols"}.issubset(keys):
        rows = d["rows"]
        cols = d["cols"]
        data = d["data"] if "data" in keys else np.ones_like(rows, dtype=np.float32)
        shape = tuple(d["shape"]) if "shape" in keys else (int(rows.max()) + 1, int(cols.max()) + 1)
        return shape, len(data)

    # formato 2: matriz sparse (arr_0)
    if "arr_0" in keys:
        arr = d["arr_0"].item() if hasattr(d["arr_0"], "item") else d["arr_0"]
        try:
            shape = tuple(arr.shape)
            n_inter = int(arr.nnz)
            return shape, n_inter
        except Exception:
            raise ValueError(f"{path} tiene 'arr_0' pero no es matriz dispersa válida")

    # formato 3: user_idx, item_idx
    if {"user_idx", "item_idx"}.issubset(keys):
        rows = d["user_idx"]
        cols = d["item_idx"]
        shape = (int(rows.max()) + 1, int(cols.max()) + 1)
        n_inter = len(rows)
        return shape, n_inter

    raise ValueError(f"Formato no reconocido para {path}: claves {sorted(keys)}")


def inspect_one(path: Path):
    shape, n = _load_triplet_npz(path)
    n_users, n_items = shape
    dens = (n / (n_users * n_items)) * 100 if n_users > 0 and n_items > 0 else 0.0
    return {
        "file": str(path),
        "users": int(n_users),
        "items": int(n_items),
        "interactions": int(n),
        "density_pct": float(dens),
        "avg_per_user": float(n / max(1, n_users)),
        "avg_per_item": float(n / max(1, n_items)),
    }


def main():
    ap = argparse.ArgumentParser(description="Inspector de splits .npz")
    ap.add_argument("--splits_dir", default="Data", help="Carpeta con train.npz/test.npz/valid.npz")
    ap.add_argument("--paths", nargs="*", help="Rutas específicas a archivos .npz (opcional)")
    ap.add_argument("--json", action="store_true", help="Imprimir también en formato JSON")
    args = ap.parse_args()

    # Buscar archivos
    paths = []
    if args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        base = Path(args.splits_dir)
        for name in ["train.npz", "test.npz", "valid.npz"]:
            p = base / name
            if p.exists():
                paths.append(p)

    if not paths:
        print("❌ No se encontraron archivos .npz. Usa --splits_dir o --paths.")
        return

    out = []
    for p in paths:
        try:
            stats = inspect_one(p)
            out.append(stats)
            print(f"📂 {stats['file']}")
            print(f"  Usuarios: {stats['users']:,}")
            print(f"  Ítems: {stats['items']:,}")
            print(f"  Interacciones: {stats['interactions']:,}")
            print(f"  Densidad: {stats['density_pct']:.6f}%")
            print(f"  Promedio interacciones por usuario: {stats['avg_per_user']:.2f}")
            print(f"  Promedio interacciones por ítem: {stats['avg_per_item']:.2f}")
            print("-" * 60)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
