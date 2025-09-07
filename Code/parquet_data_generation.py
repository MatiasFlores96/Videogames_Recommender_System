
"""
Generación de dataset canónico a partir de JSONs "no tan limpios".
- Acepta JSON válido, dicts con comillas simples/u'...', True/False, y JSON-Lines (NDJSON).
- Expande reviews anidadas (user_id -> reviews[]).
- Parseo de fecha "Posted Month Day, Year." a timestamp real.
- De-dup por (user_id, item_id) quedando la review más reciente.
- Escribe Parquet con columnas canónicas para el pipeline.

Uso (ejemplo):
  python parquet_data_generation.py \
    --interactions_json Data/raw_users_reviews.json \
    --catalog_json Data/raw_games_catalog.json \
    --out_parquet Data/dataset_procesado.parquet \
    --out_catalog_parquet Data/games_catalog.parquet
"""

import argparse, re, json, ast
from pathlib import Path
from datetime import datetime
import pandas as pd

def safe_load_json_like(path):
    """
    Carga JSON, Python literal (comillas simples/u'…'), o JSON-Lines (NDJSON).
    Devuelve lista o dict.
    """
    txt = Path(path).read_text(encoding="utf-8").strip()
    if not txt:
        return []
    # 1) Intento JSON estándar
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    # 2) Intento JSON-Lines (NDJSON)
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(lines) > 1:
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                # 3) Último recurso: literal por línea (permite '...' y True/False)
                out.append(ast.literal_eval(ln))
        return out
    # 4) Último recurso: literal del archivo completo
    return ast.literal_eval(txt)

def parse_posted(s: str):
    # ejemplos: "Posted November 5, 2011.", "Posted July 15, 2011."
    if not s:
        return None
    s = s.strip()
    m = re.search(r"Posted\s+([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})", s)
    if m:
        month, day, year = m.group(1), int(m.group(2)), int(m.group(3))
        try:
            return datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
        except ValueError:
            pass
    # fallback a ISO YYYY-MM-DD si aparece
    m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m2:
        return datetime(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
    return None

def build_interactions(users):
    """
    users: lista de dicts con shape parecido a:
      {'user_id': '...', 'user_url': '...', 'reviews': [{...}, {...}]}
    Devuelve DataFrame de interacciones canónicas.
    """
    rows = []
    for u in users if isinstance(users, list) else [users]:
        uid = u.get("user_id") or u.get("steamid") or u.get("profile_id") or u.get("userid")
        if uid is None:
            # si el objeto es directamente una review con user_id, también soportarlo
            if "reviews" not in u and "user_id" in u and "item_id" in u:
                uid = u.get("user_id")
                r = u
                rows.append({
                    "user_id": str(uid),
                    "item_id": str(r.get("item_id")),
                    "recommend": r.get("recommend", None),
                    "review_text": r.get("review", None),
                    "posted_raw": r.get("posted", None),
                    "timestamp": parse_posted(r.get("posted", "")) if r.get("posted") else None,
                    "helpful_raw": r.get("helpful", None),
                    "funny_raw": r.get("funny", None),
                })
                continue
            else:
                # no se puede deducir user_id, saltar
                continue

        reviews = u.get("reviews", [])
        for r in reviews:
            rows.append({
                "user_id": str(uid),
                "item_id": str(r.get("item_id")),
                "recommend": r.get("recommend", None),
                "review_text": r.get("review", None),
                "posted_raw": r.get("posted", None),
                "timestamp": parse_posted(r.get("posted", "")) if r.get("posted") else None,
                "helpful_raw": r.get("helpful", None),
                "funny_raw": r.get("funny", None),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # rating binario para retrieval
    df["rating"] = df["recommend"].map({True: 1.0, False: 0.0}).fillna(0.0).astype("float32")

    # completar timestamp faltantes con orden estable (evita errores del split temporal)
    if df["timestamp"].isna().any():
        df = df.sort_values(["user_id","item_id","posted_raw"]).reset_index(drop=True)
        # usar índice como fallback de orden temporal
        mask = df["timestamp"].isna()
        fallback = pd.to_datetime(pd.Series(df.index, index=df.index), unit="s")
        df.loc[mask, "timestamp"] = fallback.loc[mask].values
    # de-duplicar por usuario-item quedando la más reciente
    df = df.sort_values("timestamp").drop_duplicates(["user_id","item_id"], keep="last")

    # seleccionar columnas canónicas
    cols = ["user_id","item_id","timestamp","rating","review_text","recommend","helpful_raw","funny_raw"]
    for c in cols:
        if c not in df.columns: df[c] = None
    return df[cols]

def build_catalog(games):
    """
    games: lista de dicts del catálogo de juegos (Steam).
    Devuelve DataFrame normalizado con item_id y features.
    """
    if isinstance(games, dict):
        games = [games]
    df = pd.DataFrame(games)
    if df.empty:
        return df
    # normalizar
    if "id" in df.columns:
        df["item_id"] = df["id"].astype(str)
    elif "appid" in df.columns:
        df["item_id"] = df["appid"].astype(str)
    else:
        # intentar deducir
        for c in ["game_id","item_id"]:
            if c in df.columns:
                df["item_id"] = df[c].astype(str)
                break
    if "title" not in df.columns and "app_name" in df.columns:
        df["title"] = df["app_name"]

    # parseo de fecha de lanzamiento
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    for col in ["price","discount_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # columnas recomendadas para entrenar luego
    keep = ["item_id","title","genres","tags","publisher","developer",
            "release_date","price","discount_price","specs","early_access","url","reviews_url"]
    keep = [c for c in keep if c in df.columns] + ["item_id"]
    df = df.loc[:, sorted(set(keep))]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions_json", required=True)
    ap.add_argument("--catalog_json", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--out_catalog_parquet", default=None)
    args = ap.parse_args()

    users = safe_load_json_like(args.interactions_json)
    games = safe_load_json_like(args.catalog_json)

    df_inter = build_interactions(users)
    if df_inter.empty:
        raise SystemExit("No se construyeron interacciones. Verificá el formato del JSON de interacciones.")

    df_inter.to_parquet(args.out_parquet, engine="pyarrow", index=False)
    print("Wrote interactions parquet:", args.out_parquet, "rows:", len(df_inter))

    df_games = build_catalog(games)
    if args.out_catalog_parquet:
        df_games.to_parquet(args.out_catalog_parquet, engine="pyarrow", index=False)
        print("Wrote catalog parquet:", args.out_catalog_parquet, "rows:", len(df_games))

if __name__ == "__main__":
    main()
