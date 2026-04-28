"""
Computa embeddings diarios de sentimiento con FinBERT.

Para cada día hábil del mercado (2015-2024):
  1. Agrupa todas las noticias del día (headline + body)
  2. Extrae embedding [CLS] con FinBERT para cada noticia
  3. Promedia los embeddings del día → 1 vector de 768 dimensiones
  4. Días sin noticias: forward-fill del día hábil anterior

Reanudable: checkpoints cada CHECKPOINT_EVERY días.
Si se interrumpe, retoma desde el último checkpoint.

Requisitos:
  - pip install transformers torch tqdm
  - data/interim/corpus_merged.csv  (ejecutar build_corpus.py primero)
  - data/processed/price_df.csv     (ejecutar run_pipeline.py primero)
  - GPU recomendada (Colab T4: ~3h para 2500 días)

Output: data/processed/finbert_embeddings.csv
Schema: [date (index), emb_0, ..., emb_767]

Uso: python scripts/compute_embeddings.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROCESSED_PATH,
    FINBERT_MODEL,
    SENTIMENT_DIM,
    LOG_LEVEL,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

CORPUS_CSV       = Path(__file__).parent.parent / "data" / "interim" / "corpus_merged.csv"
PRICE_DF_CSV     = DATA_PROCESSED_PATH / "price_df.csv"
OUTPUT_CSV       = DATA_PROCESSED_PATH / "finbert_embeddings.csv"
CHECKPOINT_DIR   = DATA_PROCESSED_PATH / "emb_checkpoints"
PARTIAL_CSV      = CHECKPOINT_DIR / "partial_progress.csv"

CHECKPOINT_EVERY = 200   # guardar cada N días procesados
BATCH_CPU        = 32    # noticias por batch en CPU
BATCH_GPU        = 64    # noticias por batch en GPU T4


# ── Carga de fuentes ──────────────────────────────────────────────────────────

def load_market_days() -> pd.DatetimeIndex:
    """
    Carga los días de mercado abierto desde price_df.csv.
    Esta es la fuente de verdad para qué días necesitan embedding.
    """
    if not PRICE_DF_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró {PRICE_DF_CSV}.\n"
            "Ejecuta primero: python run_pipeline.py"
        )
    df = pd.read_csv(PRICE_DF_CSV, index_col=0, parse_dates=True)
    return df.index.normalize()


def load_corpus_index(market_days: pd.DatetimeIndex) -> dict:
    """
    Carga el corpus y lo indexa por fecha para acceso O(1).

    Returns:
        dict[pd.Timestamp, list[str]]: textos agrupados por día de mercado
    """
    if not CORPUS_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró {CORPUS_CSV}.\n"
            "Ejecuta primero: python scripts/build_corpus.py"
        )

    logger.info(f"Cargando corpus: {CORPUS_CSV}")
    df = pd.read_csv(CORPUS_CSV, parse_dates=["date"])
    df["date"]     = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["headline"] = df["headline"].fillna("").astype(str).str.strip()
    df["body"]     = df["body"].fillna("").astype(str).str.strip()
    df = df.dropna(subset=["date"])
    df = df[df["headline"] != ""]

    # Construir texto completo por noticia
    def build_text(row: pd.Series) -> str:
        return (row["headline"] + " " + row["body"]).strip()

    df["text"] = df.apply(build_text, axis=1)

    # Agrupar por día
    corpus: dict = {}
    for date, group in df.groupby("date"):
        corpus[date] = group["text"].tolist()

    logger.info(f"Corpus indexado: {len(corpus):,} días con noticias")

    # Cobertura respecto a días de mercado
    covered = sum(1 for d in market_days if d in corpus)
    logger.info(f"Días de mercado con noticias: {covered:,} / {len(market_days):,}  ({covered/len(market_days)*100:.1f}%)")

    return corpus


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Carga progreso previo desde PARTIAL_CSV (si existe)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not PARTIAL_CSV.exists():
        return {}

    logger.info(f"Reanudando desde checkpoint: {PARTIAL_CSV}")
    df = pd.read_csv(PARTIAL_CSV, index_col=0, parse_dates=True)
    results = {
        str(idx.date()): row.values.astype(np.float32)
        for idx, row in df.iterrows()
    }
    logger.info(f"  Días ya procesados: {len(results):,}")
    return results


def save_checkpoint(results: dict) -> None:
    """Guarda progreso parcial en PARTIAL_CSV."""
    emb_cols = [f"emb_{i}" for i in range(SENTIMENT_DIM)]
    rows = {k: v for k, v in results.items()}

    df = pd.DataFrame.from_dict(rows, orient="index", columns=emb_cols)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.to_csv(PARTIAL_CSV)
    logger.info(f"Checkpoint guardado: {len(results):,} días → {PARTIAL_CSV.name}")


# ── Cómputo de embeddings ─────────────────────────────────────────────────────

def compute_day_embedding(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """
    Computa embedding promedio de todas las noticias de un día.

    Proceso por noticia:
      - Tokenizar (max 512 tokens, truncar si excede)
      - Extraer embedding [CLS] de la última capa de FinBERT
    Proceso de agregación diaria:
      - Promediar embeddings de todas las noticias del día

    Returns:
        ndarray (768,) — embedding promedio del día
    """
    all_cls = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        # Token [CLS] = índice 0 de la secuencia
        cls_emb = output.last_hidden_state[:, 0, :].cpu().float().numpy()
        all_cls.append(cls_emb)

    embeddings = np.vstack(all_cls)          # (n_noticias, 768)
    return embeddings.mean(axis=0)           # (768,)


# ── Pipeline principal ────────────────────────────────────────────────────────

def main() -> int:
    if OUTPUT_CSV.exists():
        logger.info(f"Embeddings ya existen: {OUTPUT_CSV}")
        logger.info("Para recomputar, elimina el archivo y vuelve a ejecutar.")
        return 0

    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        logger.error("Instala transformers: pip install transformers")
        return 1

    # ── Setup ─────────────────────────────────────────────────────────────────
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = BATCH_GPU if device == "cuda" else BATCH_CPU

    logger.info(f"Dispositivo  : {device}")
    if device == "cuda":
        logger.info(f"GPU          : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Batch size   : {batch_size}")
    logger.info(f"FinBERT      : {FINBERT_MODEL}")

    logger.info("Cargando tokenizador y modelo FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    finbert   = AutoModel.from_pretrained(FINBERT_MODEL).to(device)
    finbert.eval()
    logger.info("Modelo listo.")

    # ── Datos ────────────────────────────────────────────────────────────────
    market_days = load_market_days()
    corpus      = load_corpus_index(market_days)
    results     = load_checkpoint()

    # Días aún no procesados
    pending = [d for d in market_days if str(d.date()) not in results]
    logger.info(f"Días pendientes: {len(pending):,}")

    if not pending:
        logger.info("Todos los días ya están procesados. Generando CSV final...")
    else:
        # ── Cómputo ──────────────────────────────────────────────────────────
        last_valid: np.ndarray | None = (
            next(iter(results.values())) if results else None
        )
        n_real, n_filled, n_since_ckpt = 0, 0, 0

        for day in tqdm(pending, desc="FinBERT embeddings", unit="día"):
            day_ts  = pd.Timestamp(day).normalize()
            day_key = str(day_ts.date())

            # Buscar noticias del día (±1d para desfases de zona horaria)
            texts = (
                corpus.get(day_ts) or
                corpus.get(day_ts - pd.Timedelta("1d")) or
                corpus.get(day_ts + pd.Timedelta("1d"))
            )

            if texts:
                emb = compute_day_embedding(texts, tokenizer, finbert, device, batch_size)
                last_valid = emb
                n_real += 1
            elif last_valid is not None:
                # Forward-fill: sentimiento persiste si no hay noticias
                emb = last_valid
                n_filled += 1
            else:
                # Solo al inicio absoluto (sin historial previo)
                emb = np.zeros(SENTIMENT_DIM, dtype=np.float32)
                n_filled += 1

            results[day_key] = emb.astype(np.float32)
            n_since_ckpt += 1

            if n_since_ckpt >= CHECKPOINT_EVERY:
                save_checkpoint(results)
                n_since_ckpt = 0

        # Checkpoint final antes de guardar
        if n_since_ckpt > 0:
            save_checkpoint(results)

        logger.info(f"Días con noticias reales : {n_real:,}")
        logger.info(f"Días forward-filled      : {n_filled:,}")

    # ── Construir y guardar CSV final ─────────────────────────────────────────
    emb_cols = [f"emb_{i}" for i in range(SENTIMENT_DIM)]

    rows = []
    for day in market_days:
        key = str(pd.Timestamp(day).date())
        if key in results:
            rows.append([day] + results[key].tolist())

    if not rows:
        logger.error("No se pudo construir el CSV — ningún día procesado correctamente.")
        return 1

    df_out = pd.DataFrame(rows, columns=["date"] + emb_cols)
    df_out = df_out.set_index("date")
    df_out.index = pd.to_datetime(df_out.index)

    DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV)

    # ── Sanity check: COVID crash ─────────────────────────────────────────────
    logger.info("\n[SANITY CHECK] Embedding COVID-19 (2020-03-16):")
    covid_key = pd.Timestamp("2020-03-16")
    if covid_key in df_out.index:
        emb_covid = df_out.loc[covid_key].values
        norm = np.linalg.norm(emb_covid)
        logger.info(f"  Norma del vector : {norm:.4f}")
        if norm < 1.0:
            logger.warning("  ⚠️  Norma anormalmente baja — posible problema de alineación de fechas")
            logger.warning("     El crash COVID debería producir un embedding de alta magnitud")
        else:
            logger.info("  ✓  Norma dentro del rango esperado")
    else:
        logger.warning("  ⚠️  Fecha 2020-03-16 no encontrada en el índice")

    # ── Reporte final ─────────────────────────────────────────────────────────
    norms = np.linalg.norm(df_out.values, axis=1)

    logger.info(f"\n{'='*65}")
    logger.info("EMBEDDINGS FINBERT COMPLETADOS")
    logger.info(f"{'='*65}")
    logger.info(f"  Archivo          : {OUTPUT_CSV}")
    logger.info(f"  Shape            : {df_out.shape}  (días × 768 dims)")
    logger.info(f"  Norma media      : {norms.mean():.4f}")
    logger.info(f"  Norma mín / máx  : {norms.min():.4f}  /  {norms.max():.4f}")
    logger.info(f"{'='*65}")
    logger.info("\n✅ FinBERT embeddings listos.")
    logger.info("Próximo:")
    logger.info("  1. python run_pipeline.py          (regenera price_seqs.npy con sentimiento real)")
    logger.info("  2. Abrir QQQ_Hibrido_Completo.ipynb y ejecutar todas las celdas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
