"""
Computa embeddings diarios de sentimiento con FinBERT.

Para cada día hábil del mercado (período de config.py, por defecto 2015-2023):
  1. Agrupa todas las noticias del día (headline + body)
  2. Extrae embedding [CLS] con FinBERT para cada noticia
  3. Promedia los embeddings del día → 1 vector de 768 dimensiones
  4. Días sin noticias: forward-fill del día hábil anterior

El corpus NO se carga entero en memoria: llega ordenado por fecha desde
build_corpus.py, así que se recorre por chunks manteniendo solo una ventana de
±1 día (lo que necesita el fallback de zona horaria). Con millones de titulares,
cargarlo como dict reventaba la RAM de Colab.

Reanudable: checkpoints cada CHECKPOINT_EVERY días.
Si se interrumpe, retoma desde el último checkpoint.

Requisitos:
  - pip install transformers torch tqdm
  - data/interim/corpus_merged.csv  (ejecutar build_corpus.py primero)
  - data/processed/price_df.csv     (ejecutar run_pipeline.py primero)
  - GPU recomendada (Colab T4)

Variables de entorno opcionales:
  FINBERT_FP16=0        desactiva la inferencia en fp16 (por defecto activa en
                        GPU; ~2-3x más rápida, diferencia numérica despreciable
                        al promediar cientos de noticias por día)
  MAX_NEWS_PER_DAY=N    submuestrea a N noticias por día. Por defecto 0 (sin
                        límite). OJO: activarlo es una decisión metodológica,
                        no solo de rendimiento — cambia qué entra al promedio.

Output: data/processed/finbert_embeddings.csv
Schema: [date (index), emb_0, ..., emb_767]

Uso: python scripts/compute_embeddings.py
"""

import logging
import os
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

CHECKPOINT_EVERY = 200      # guardar cada N días procesados
BATCH_CPU        = 32       # noticias por batch en CPU
BATCH_GPU        = 64       # noticias por batch en GPU T4
CORPUS_CHUNKSIZE = 200_000  # filas por chunk al recorrer el corpus

ONE_DAY          = pd.Timedelta("1d")
MAX_NEWS_PER_DAY = int(os.getenv("MAX_NEWS_PER_DAY", "0"))


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


def iter_corpus_days(path: Path):
    """
    Genera (fecha, textos) en orden cronológico recorriendo el corpus por chunks.

    Requiere que el CSV venga ordenado por fecha — build_corpus.py lo garantiza.
    Un día puede quedar partido entre dos chunks, así que se arrastra el grupo
    abierto hasta confirmar que cambió la fecha.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}.\n"
            "Ejecuta primero: python scripts/build_corpus.py"
        )

    open_date: pd.Timestamp | None = None
    open_texts: list[str] = []

    reader = pd.read_csv(path, chunksize=CORPUS_CHUNKSIZE, low_memory=False)
    for chunk in reader:
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.normalize()
        chunk = chunk.dropna(subset=["date"])

        chunk["headline"] = chunk["headline"].fillna("").astype(str).str.strip()
        chunk["body"]     = chunk["body"].fillna("").astype(str).str.strip()
        chunk = chunk[chunk["headline"] != ""]
        if chunk.empty:
            continue

        # Vectorizado — df.apply(axis=1) era fila a fila en Python puro y a
        # escala de millones de titulares dominaba el tiempo de esta función.
        chunk["text"] = (chunk["headline"] + " " + chunk["body"]).str.strip()

        for date, group in chunk.groupby("date", sort=True):
            texts = group["text"].tolist()

            if open_date is None:
                open_date, open_texts = date, texts
            elif date == open_date:
                open_texts.extend(texts)
            elif date < open_date:
                raise ValueError(
                    f"El corpus no está ordenado por fecha ({date.date()} aparece "
                    f"después de {open_date.date()}).\n"
                    "Reconstrúyelo: elimina corpus_merged.csv y corre build_corpus.py"
                )
            else:
                yield open_date, open_texts
                open_date, open_texts = date, texts

    if open_date is not None:
        yield open_date, open_texts


class CorpusWindow:
    """
    Ventana deslizante sobre el corpus: mantiene en RAM solo los días cercanos al
    día de mercado que se está procesando (necesarios para el fallback de ±1d).
    """

    def __init__(self, path: Path):
        self._iter = iter_corpus_days(path)
        self._pending: tuple | None = None
        self._exhausted = False
        self._window: dict = {}
        self.days_seen = 0

    def _advance_to(self, limit: pd.Timestamp) -> None:
        """Consume el corpus hasta pasar `limit`, guardando lo leído en la ventana."""
        while not self._exhausted:
            if self._pending is None:
                self._pending = next(self._iter, None)
                if self._pending is None:
                    self._exhausted = True
                    break
                self.days_seen += 1

            date, texts = self._pending
            if date > limit:
                break

            self._window[date] = texts
            self._pending = None

    def texts_for(self, day: pd.Timestamp) -> list[str] | None:
        """Textos del día, con el mismo fallback ±1d del diseño original."""
        self._advance_to(day + ONE_DAY)

        # Purgar lo que ya no se volverá a consultar
        for stale in [d for d in self._window if d < day - ONE_DAY]:
            del self._window[stale]

        return (
            self._window.get(day) or
            self._window.get(day - ONE_DAY) or
            self._window.get(day + ONE_DAY)
        )


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

    df = pd.DataFrame.from_dict(dict(results), orient="index", columns=emb_cols)
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
    use_fp16: bool,
) -> np.ndarray:
    """
    Computa embedding promedio de todas las noticias de un día.

    Proceso por noticia:
      - Tokenizar (max 512 tokens, truncar si excede)
      - Extraer embedding [CLS] de la última capa de FinBERT
    Proceso de agregación diaria:
      - Promediar embeddings de todas las noticias del día

    Los textos se ordenan por longitud antes de agrupar en batches: así cada
    batch se rellena hasta una longitud parecida y se desperdicia mucho menos
    cómputo en padding. El promedio no depende del orden, así que es seguro.

    Returns:
        ndarray (768,) — embedding promedio del día
    """
    ordered = sorted(texts, key=len)

    # Suma corrida en vez de apilar todo: un día con miles de noticias no tiene
    # por qué materializar una matriz (n_noticias, 768).
    total = np.zeros(SENTIMENT_DIM, dtype=np.float64)
    seen  = 0

    for i in range(0, len(ordered), batch_size):
        batch = ordered[i : i + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(**encoded)
            else:
                output = model(**encoded)

        # Token [CLS] = índice 0 de la secuencia
        cls_emb = output.last_hidden_state[:, 0, :].float().cpu().numpy()
        total += cls_emb.sum(axis=0)
        seen  += len(batch)

    return (total / seen).astype(np.float32)


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
    use_fp16   = device == "cuda" and os.getenv("FINBERT_FP16", "1") != "0"

    logger.info(f"Dispositivo  : {device}")
    if device == "cuda":
        logger.info(f"GPU          : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Batch size   : {batch_size}")
    logger.info(f"Precisión    : {'fp16 (autocast)' if use_fp16 else 'fp32'}")
    logger.info(f"FinBERT      : {FINBERT_MODEL}")
    if MAX_NEWS_PER_DAY:
        logger.warning(
            f"MAX_NEWS_PER_DAY={MAX_NEWS_PER_DAY}: se submuestrean las noticias de "
            "cada día. Documenta esto como decisión metodológica en la tesis."
        )

    logger.info("Cargando tokenizador y modelo FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    finbert   = AutoModel.from_pretrained(FINBERT_MODEL).to(device)
    finbert.eval()
    logger.info("Modelo listo.")

    # ── Datos ────────────────────────────────────────────────────────────────
    market_days = load_market_days()
    results     = load_checkpoint()
    window      = CorpusWindow(CORPUS_CSV)

    n_done_before = len(results)
    n_pending = sum(1 for d in market_days if str(pd.Timestamp(d).date()) not in results)
    logger.info(f"Días de mercado : {len(market_days):,}")
    logger.info(f"Días pendientes : {n_pending:,}")

    rng = np.random.default_rng(42)   # submuestreo reproducible si hay tope
    n_real, n_filled, n_since_ckpt = 0, 0, 0
    last_valid: np.ndarray | None = next(iter(results.values())) if results else None

    # Se recorren TODOS los días de mercado aunque haya checkpoint: la ventana
    # del corpus avanza en orden y debe mantenerse sincronizada con el calendario.
    for day in tqdm(market_days, desc="FinBERT embeddings", unit="día"):
        day_ts  = pd.Timestamp(day).normalize()
        day_key = str(day_ts.date())

        texts = window.texts_for(day_ts)

        if day_key in results:
            last_valid = results[day_key]
            continue

        if texts:
            if MAX_NEWS_PER_DAY and len(texts) > MAX_NEWS_PER_DAY:
                idx = rng.choice(len(texts), MAX_NEWS_PER_DAY, replace=False)
                texts = [texts[i] for i in idx]

            emb = compute_day_embedding(
                texts, tokenizer, finbert, device, batch_size, use_fp16
            )
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

    if n_since_ckpt > 0:
        save_checkpoint(results)

    logger.info(f"Días del corpus leídos   : {window.days_seen:,}")
    logger.info(f"Días con noticias reales : {n_real:,}")
    logger.info(f"Días forward-filled      : {n_filled:,}")
    if n_done_before:
        logger.info(f"Días reusados del checkpoint: {n_done_before:,}")

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
