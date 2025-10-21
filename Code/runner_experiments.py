#!/usr/bin/env python3
"""
Runner de experimentos (desde cero) para tu tesis.

Características clave
---------------------
- Carga `train.npz`/`test.npz` (y opcional `valid.npz`) desde `--splits_dir`.
- Soporta múltiples formatos de .npz: (rows, cols, data, shape) | arr_0 con csr | (user_idx, item_idx, label).
- Importa modelos dinámicamente desde `Code.Models.<Nombre>` y los instancia con hiperparámetros.
- Protocolo Top-N con métricas **Recall@K** y **NDCG@K** (sin coverage/diversity).
- **Grid de hiperparámetros por CLI** (`--grid`) y **selección automática del mejor** en valid por `--primary_metric`.
- Re-entrena el mejor en train y lo evalúa en test. Guarda CSVs en `--results_dir`.

Requisitos de cada modelo (adapter)
-----------------------------------
Debe exponer una clase con:
    class Recommender(...):
        def fit(self, train_csr): ...
        def recommend(self, user_id:int, k:int, exclude_seen=None) -> List[int]

Uso rápido
---------
python -m Code.runner_experiments \
  --splits_dir Data \
  --results_dir Results \
  --topk 10 20 \
  --models UserUser_CF ItemItem_CF \
  --grid UserUser_CF:K=40,80,150 ItemItem_CF:K=40,80,150 \
  --valid_npz Data/valid.npz \
  --primary_metric NDCG@10 \
  --verbose
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy import sparse as sp

# ===============================
# Utilidades: métricas (Recall@K, NDCG@K)
# ===============================

def recall_at_k(recommended: List[int], relevant: set[int], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    hits = sum(1 for i in recommended[:k] if i in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    # DCG
    dcg = 0.0
    for rank, iid in enumerate(recommended[:k], start=1):
        if iid in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    # IDCG
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg

# ===============================
# Carga splits .npz en distintos formatos
# ===============================

def load_npz_triplet(path: Path) -> sp.csr_matrix:
    """Admite tres formatos de npz:
    1) rows, cols, data, shape
    2) arr_0 con matriz scipy.sparse
    3) user_idx, item_idx, (label)
    """
    data = np.load(path, allow_pickle=True)
    keys = set(data.keys())

    # 1) tripletas clásicas
    if {"rows", "cols"}.issubset(keys):
        rows = data["rows"].astype(int)
        cols = data["cols"].astype(int)
        vals = data.get("data")
        if vals is None:
            vals = np.ones_like(rows, dtype=np.float32)
        shape = tuple(data["shape"]) if "shape" in keys else (int(rows.max()) + 1, int(cols.max()) + 1)
        return sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()

    # 2) matriz guardada entera
    if "arr_0" in keys:
        mat = data["arr_0"].item() if hasattr(data["arr_0"], "item") else data["arr_0"]
        if isinstance(mat, sp.spmatrix):
            return mat.tocsr()
        raise ValueError(f"{path} tiene 'arr_0' pero no es scipy.sparse")

    # 3) tripletas user/item/label
    if {"user_idx", "item_idx"}.issubset(keys):
        rows = data["user_idx"].astype(int)
        cols = data["item_idx"].astype(int)
        vals = data["label"].astype(float) if "label" in keys else np.ones_like(rows, dtype=np.float32)
        shape = (int(rows.max()) + 1, int(cols.max()) + 1)
        return sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()

    raise ValueError(f"Formato no reconocido para {path}, claves: {list(keys)}")

# ===============================
# Adaptador dinámico de modelos
# ===============================

@dataclass
class ModelSpec:
    module_name: str
    class_name: Optional[str] = None
    params: Dict = None

class ModelAdapter:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.model = None

    def load(self):
        mod = importlib.import_module(self.spec.module_name)
        cls = None
        if self.spec.class_name and hasattr(mod, self.spec.class_name):
            cls = getattr(mod, self.spec.class_name)
        else:
            # heurística: buscar clase con fit+recommend o fit+predict_scores
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                methods = {m for m, _ in inspect.getmembers(obj, inspect.isfunction)}
                if {"fit", "recommend"}.issubset(methods) or {"fit", "predict_scores"}.issubset(methods):
                    cls = obj
                    break
        if cls is None:
            raise RuntimeError(
                f"No se encontró una clase con fit/recommend en {self.spec.module_name}. "
                f"Indicá class_name o añadí un adapter Recommender."
            )
        params = self.spec.params or {}
        self.model = cls(**params)

    def fit(self, train_csr: sp.csr_matrix):
        try:
            return self.model.fit(train_csr)
        except TypeError:
            return self.model.fit(X=train_csr)

    def recommend_user(self, user_id: int, k: int, exclude_seen: Optional[sp.csr_matrix] = None) -> List[int]:
        # Método recommend preferido
        if hasattr(self.model, "recommend"):
            try:
                recs = self.model.recommend(user_id, k=k, exclude_seen=exclude_seen)
                return list(map(int, recs))
            except TypeError:
                recs = self.model.recommend(user_id, k)
                return list(map(int, recs))
        # Alternativa: predict_scores
        if hasattr(self.model, "predict_scores"):
            scores = self.model.predict_scores(user_id)
            if exclude_seen is not None:
                seen = exclude_seen[user_id].indices
                scores[seen] = -np.inf
            top = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
            top = top[np.argsort(-scores[top])]
            return list(map(int, top))
        raise RuntimeError("Modelo sin recommend ni predict_scores compatible.")

# ===============================
# Evaluación Top-N
# ===============================

def evaluate_model(adapter: ModelAdapter,
                   train: sp.csr_matrix,
                   test: sp.csr_matrix,
                   topk: List[int],
                   eval_users: Optional[Iterable[int]] = None,
                   seed: int = 42,
                   verbose: bool = False) -> Dict[str, float]:
    random.seed(seed); np.random.seed(seed)

    adapter.fit(train)
    n_users_train, _ = train.shape
    n_users_test,  _ = test.shape
    if eval_users is None:
        users = list(range(min(n_users_train, n_users_test)))
    else:
        users = [u for u in eval_users if u < n_users_train and u < n_users_test]

    test_csr = test.tocsr()
    relevant_per_user = {u: set(test_csr[u].indices.tolist()) for u in users}

    results = {f"Recall@{k}": [] for k in topk}
    results.update({f"NDCG@{k}": [] for k in topk})

    Kmax = max(topk)
    for u in users:
        recs = adapter.recommend_user(u, k=Kmax, exclude_seen=train)
        rel = relevant_per_user[u]
        for k in topk:
            results[f"Recall@{k}"].append(recall_at_k(recs, rel, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(recs, rel, k))

    summary = {f"Recall@{k}": float(np.mean(results[f"Recall@{k}"])) for k in topk}
    summary.update({f"NDCG@{k}": float(np.mean(results[f"NDCG@{k}"])) for k in topk})
    return summary

# ===============================
# Helpers: grid y selección del mejor
# ===============================

def parse_grid(grid_args: List[str]) -> Dict[str, Dict[str, List]]:
    """Convierte
    ['UserUser_CF:K=40,80', 'ItemItem_CF:K=40,80', 'UserUser_CF:weighting=none']
    -> {'UserUser_CF': {'K':[40,80], 'weighting':['none']}, 'ItemItem_CF': {'K':[40,80]}}
    """
    out: Dict[str, Dict[str, List]] = {}
    for token in grid_args:
        model, kv = token.split(":", 1)
        k, vals = kv.split("=", 1)
        lst = []
        for v in vals.split(","):
            v = v.strip()
            # intentar castear
            try:
                v = int(v)
            except:
                try:
                    v = float(v)
                except:
                    pass
            lst.append(v)
        out.setdefault(model, {}).setdefault(k, lst)
    return out


def combos_from(d: Dict[str, List]) -> List[Dict]:
    if not d:
        return [ {} ]
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    return [ {k: v for k, v in zip(keys, tup)} for tup in product(*vals) ]


def select_best(rows: List[Dict], primary_metric: str = "NDCG@10") -> Dict:
    """Selecciona el mejor por métrica principal; desempata con Recall@10."""
    def get(row, key):
        x = row.get(key, float("nan"))
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return -float("inf")
        return x
    best = None
    for r in rows:
        cand = (get(r, primary_metric), get(r, "Recall@10"))
        if best is None:
            best = r
        else:
            if cand > (get(best, primary_metric), get(best, "Recall@10")):
                best = r
    return best

# ===============================
# MAIN
# ===============================

def main():
    parser = argparse.ArgumentParser(description="Runner de experimentos Top-N (Recall/NDCG)")
    parser.add_argument("--splits_dir", type=str, default="Data", help="Carpeta con train.npz/test.npz y opcional valid.npz")
    parser.add_argument("--valid_npz", type=str, default=None, help="Ruta a valid.npz (opcional)")
    parser.add_argument("--results_dir", type=str, default="Results", help="Carpeta de salida")
    parser.add_argument("--models", nargs="+", required=True, help="Nombres bajo Code.Models, ej: UserUser_CF ItemItem_CF")
    parser.add_argument("--topk", nargs="+", type=int, default=[10], help="Ks para Recall/NDCG")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid", nargs="*", default=[], help="Grid por modelo: Model:param=v1,v2 ...")
    parser.add_argument("--primary_metric", type=str, default="NDCG@10", help="Métrica para elegir el mejor en valid")
    parser.add_argument("--eval_users_max", type=int, default=None, help="Subsample de usuarios para debug/velocidad")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    np.random.seed(args.seed); random.seed(args.seed)

    splits_dir = Path(args.splits_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("[runner] Python:", sys.version)
        print("[runner] CWD:", os.getcwd())
        print("[runner] splits_dir:", splits_dir)

    # Cargar splits
    train = load_npz_triplet(splits_dir / "train.npz")
    test  = load_npz_triplet(splits_dir / "test.npz")
    valid = load_npz_triplet(Path(args.valid_npz)) if args.valid_npz else None

    # --- Alinear valid/test a la forma de train (usuarios e items) ---
    def align_to_train(mat, train):
        """Recorta mat para que no exceda filas/columnas de train."""
        if mat is None:
            return None
        mat = mat.tocsr()
        nu, ni = train.shape
        mu, mi = mat.shape
        if mu > nu:
            mat = mat[:nu, :]
        if mi > ni:
            mat = mat[:, :ni]
        return mat

    test  = align_to_train(test, train)
    valid = align_to_train(valid, train) if valid is not None else None

    # Usuarios a evaluar
    eval_users = None
    if args.eval_users_max is not None:
        n_users = train.shape[0]
        users = list(range(n_users))
        random.shuffle(users)
        eval_users = users[: args.eval_users_max]

    # Preparar modelos + grids
    def to_spec(name: str, params: Dict | None = None) -> ModelSpec:
        return ModelSpec(module_name=f"Code.Models.{name}", class_name=None, params=params or {})

    model_specs = [to_spec(m) for m in args.models]
    grids = parse_grid(args.grid)  # {'UserUser_CF': {'K':[40,80]}, ...}

    # Metadatos de corrida
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_meta = {
        "timestamp": stamp,
        "seed": args.seed,
        "topk": args.topk,
        "models": args.models,
        "splits_dir": str(splits_dir),
        "valid_npz": args.valid_npz or "",
        "primary_metric": args.primary_metric,
    }

    rows_valid: List[Dict] = []  # todas las combinaciones en valid
    rows_out:   List[Dict] = []  # resultados finales en test

    for spec in model_specs:
        model_name = spec.module_name.split(".")[-1]
        grid_for_model = grids.get(model_name, {})
        combos = combos_from(grid_for_model)  # [{}] si no hay grid

        per_model_valid: List[Dict] = []
        for params in combos:
            spec.params = params
            if args.verbose:
                print(f"\n>>> {spec.module_name} params={params}")

            # instanciar y fit
            try:
                adapter = ModelAdapter(spec)
                adapter.load()
                t0 = time.time()
                adapter.fit(train)
                fit_time = time.time() - t0
            except Exception as e:
                print(f"[ERROR] {spec.module_name} (fit/load): {e}")
                continue

            # Si hay valid: evaluar para seleccionar mejor
            if valid is not None:
                try:
                    res_v = evaluate_model(adapter, train=train, test=valid, topk=args.topk, eval_users=eval_users, seed=args.seed)
                    row_v = {"model": spec.module_name, **params, **res_v, "split": "valid", "time_sec": fit_time}
                    rows_valid.append(row_v); per_model_valid.append(row_v)
                    if args.verbose:
                        print("[valid]", row_v)
                except Exception as e:
                    print(f"[ERROR] {spec.module_name} (valid): {e}")
                    continue
            else:
                # Sin valid: evaluar directo en test
                try:
                    res_t = evaluate_model(adapter, train=train, test=test, topk=args.topk, eval_users=eval_users, seed=args.seed)
                    row_t = {"model": spec.module_name, **params, **res_t, "split": "test", "time_sec": fit_time}
                    rows_out.append(row_t)
                    if args.verbose:
                        print("[test]", row_t)
                except Exception as e:
                    print(f"[ERROR] {spec.module_name} (test): {e}")
                    continue

        # Si hubo valid, elegir mejor y testear
        if valid is not None and per_model_valid:
            best = select_best(per_model_valid, primary_metric=args.primary_metric)
            if args.verbose:
                print("[best on valid]", best)
            best_params = {k: best[k] for k in best.keys() if k not in {"model", "split", "time_sec", *[f"Recall@{k}" for k in args.topk], *[f"NDCG@{k}" for k in args.topk]}}
            spec.params = best_params
            try:
                adapter = ModelAdapter(spec); adapter.load()
                t0 = time.time(); adapter.fit(train); fit_time = time.time() - t0
                res_t = evaluate_model(adapter, train=train, test=test, topk=args.topk, eval_users=eval_users, seed=args.seed)
                row_t = {"model": spec.module_name, **best_params, **res_t, "split": "test", "time_sec": fit_time}
                rows_out.append(row_t)
                if args.verbose:
                    print("[test with best]", row_t)
            except Exception as e:
                print(f"[ERROR] {spec.module_name} (best->test): {e}")
                continue

    # Guardar resultados
    import pandas as pd
    df_test = pd.DataFrame(rows_out)
    test_csv = results_dir / f"summary_{stamp}.csv"
    df_test.to_csv(test_csv, index=False)

    if rows_valid:
        df_valid = pd.DataFrame(rows_valid)
        valid_csv = results_dir / f"valid_{stamp}.csv"
        df_valid.to_csv(valid_csv, index=False)

    # Meta
    with open(results_dir / f"meta_{stamp}.json", "w", encoding="utf-8") as f:
        import json
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print("Guardado:", test_csv)
    if rows_valid:
        print("Guardado:", valid_csv)


if __name__ == "__main__":
    # Permitir ejecutar desde raíz del repo
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    main()
