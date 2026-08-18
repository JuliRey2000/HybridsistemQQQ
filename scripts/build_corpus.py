"""
Unifica y limpia el corpus de noticias (FNSPID + Tiingo 2024).

Pasos:
  1. Recorre fnspid_news.csv + tiingo_2024.csv por chunks
  2. Reparte las filas en buckets temporales por año-mes
  3. Por bucket: deduplica por (date, headline) y ordena cronológicamente
  4. Concatena los buckets en orden → data/interim/corpus_merged.csv

Por qué buckets y no pd.read_csv() directo: FNSPID deja millones de titulares y
cargarlo entero revienta la RAM de Colab. El truco es que dos filas duplicadas
por (date, headline) comparten fecha, así que siempre caen en el mismo bucket —
deduplicar dentro de cada bucket es equivalente al drop_duplicates global, pero
con memoria acotada al bucket más grande (un mes).

La salida queda ORDENADA POR FECHA a propósito: es lo que permite que
compute_embeddings.py la recorra día por día sin cargarla entera.

Output: data/interim/corpus_merged.csv
Schema: [date, headline, body]

Uso: python scripts/build_corpus.py
"""

import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW_PATH, START_DATE, END_DATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

INTERIM_DIR  = Path(__file__).parent.parent / "data" / "interim"
FNSPID_CSV   = DATA_RAW_PATH / "fnspid_news.csv"
TIINGO_CSV   = DATA_RAW_PATH / "tiingo_2024.csv"
OUTPUT_CSV   = INTERIM_DIR / "corpus_merged.csv"
BUCKET_DIR   = INTERIM_DIR / "_buckets"

CHUNKSIZE    = 500_000   # filas por chunk al leer las fuentes
COLUMNS      = ["date", "headline", "body"]


def normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Limpia un chunk al esquema estándar [date, headline, body]."""
    df = chunk.copy()

    # Las fuentes ya escriben fechas ISO normalizadas; ruta rápida con fallback.
    s = df["date"].astype(str).str.strip()
    parsed = pd.to_datetime(s.str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    failed = parsed.isna() & s.ne("") & s.ne("nan")
    if failed.any():
        parsed.loc[failed] = pd.to_datetime(s.loc[failed], errors="coerce")
    df["date"] = parsed.dt.normalize()

    df = df.dropna(subset=["date", "headline"])

    df["headline"] = df["headline"].astype(str).str.strip()
    if "body" not in df.columns:
        df["body"] = ""
    df["body"] = df["body"].fillna("").astype(str).str.strip()

    df = df[df["headline"] != ""]
    return df[COLUMNS]


def scatter_source(path: Path, label: str, writers: dict, stats: dict) -> None:
    """
    Recorre una fuente por chunks y reparte sus filas en buckets por año-mes.
    Los buckets son CSV temporales; nada se acumula en RAM.
    """
    if not path.exists():
        logger.warning(f"{label} no encontrado en {path}. Omitiendo esta fuente.")
        return

    logger.info(f"Recorriendo {label}: {path.name}")
    n_rows = 0

    reader = pd.read_csv(path, chunksize=CHUNKSIZE, low_memory=False, on_bad_lines="skip")
    for chunk in reader:
        df = normalize_chunk(chunk)
        if df.empty:
            continue

        n_rows += len(df)

        # Repartir por año-mes. La clave sale de la fecha, así que los duplicados
        # por (date, headline) caen forzosamente en el mismo bucket.
        for key, group in df.groupby(df["date"].dt.strftime("%Y-%m"), sort=False):
            if key not in writers:
                fh = open(BUCKET_DIR / f"{key}.csv", "w", encoding="utf-8", newline="")
                writers[key] = {"fh": fh, "header": True}

            w = writers[key]
            group.to_csv(w["fh"], header=w["header"], index=False)
            w["header"] = False

    stats["per_source"][label] = n_rows
    logger.info(f"  {label:12s}: {n_rows:,} noticias válidas repartidas en buckets")


def gather_buckets(writers: dict, stats: dict) -> None:
    """
    Segunda pasada: por cada bucket en orden cronológico, deduplica, ordena y
    lo anexa a la salida final. Solo un mes vive en RAM a la vez.
    """
    logger.info(f"Consolidando {len(writers)} buckets mensuales...")

    first_write = True
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as out:
        for key in sorted(writers):
            bucket_path = BUCKET_DIR / f"{key}.csv"
            df = pd.read_csv(bucket_path, parse_dates=["date"], low_memory=False)
            df["body"] = df["body"].fillna("").astype(str)

            n_before = len(df)
            df = df.drop_duplicates(subset=["date", "headline"], keep="first")
            stats["dupes"] += n_before - len(df)

            df = df.sort_values("date")

            stats["kept"] += len(df)
            stats["by_year"].update(df["date"].dt.year.tolist())
            stats["days"].update(df["date"].tolist())

            df.to_csv(out, header=first_write, index=False)
            first_write = False

    if first_write:
        raise RuntimeError("No se escribió ninguna fila al corpus final.")


def print_report(stats: dict) -> None:
    days = sorted(stats["days"])
    business_days = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    coverage_pct = len(days) / len(business_days) * 100

    logger.info(f"\n{'='*60}")
    logger.info("REPORTE CORPUS MERGED")
    logger.info(f"{'='*60}")
    for label, n in stats["per_source"].items():
        logger.info(f"  {label:22s}: {n:,}")
    logger.info(f"  Duplicados eliminados  : {stats['dupes']:,}")
    logger.info(f"  Total noticias         : {stats['kept']:,}")
    logger.info(f"  Días únicos con noticias: {len(days):,}")
    logger.info(f"  Cobertura (aprox)      : {coverage_pct:.1f}%  de días hábiles {START_DATE[:4]}-{END_DATE[:4]}")
    logger.info(f"  Rango final            : {days[0].date()} → {days[-1].date()}")

    logger.info("  Noticias por año:")
    for year in sorted(stats["by_year"]):
        logger.info(f"    {year}: {stats['by_year'][year]:8,}")

    avg_per_day = stats["kept"] / len(days) if days else 0
    logger.info(f"  Media noticias/día     : {avg_per_day:,.0f}")
    logger.info(f"{'='*60}")

    # El costo de compute_embeddings.py escala con esto — avisar si es enorme.
    if stats["kept"] > 3_000_000:
        logger.warning(
            f"\n⚠️  El corpus tiene {stats['kept']:,} noticias. compute_embeddings.py hace "
            f"una pasada de FinBERT por cada una:\n"
            f"    estima varias horas en T4 incluso con fp16. El script guarda checkpoints "
            f"cada 200 días, así que puede reanudarse si Colab se desconecta."
        )


def main() -> int:
    if OUTPUT_CSV.exists():
        logger.info(f"Corpus ya existe: {OUTPUT_CSV}")
        logger.info("Para reconstruir, elimina el archivo y vuelve a ejecutar.")
        return 0

    if not FNSPID_CSV.exists() and not TIINGO_CSV.exists():
        logger.error(
            "No se encontraron fuentes de noticias.\n"
            "Ejecuta primero:\n"
            "  python scripts/download_fnspid.py\n"
            "  python scripts/download_tiingo.py"
        )
        return 1

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(BUCKET_DIR, ignore_errors=True)   # restos de una corrida abortada
    BUCKET_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"per_source": {}, "kept": 0, "dupes": 0, "by_year": Counter(), "days": set()}
    writers: dict = {}

    try:
        scatter_source(FNSPID_CSV, "FNSPID", writers, stats)
        scatter_source(TIINGO_CSV, "Tiingo", writers, stats)

        # Cerrar los buckets antes de releerlos
        for w in writers.values():
            w["fh"].close()

        if not writers:
            logger.error("Las fuentes existen pero no produjeron ninguna fila válida.")
            return 1

        gather_buckets(writers, stats)
    finally:
        for w in writers.values():
            if not w["fh"].closed:
                w["fh"].close()
        shutil.rmtree(BUCKET_DIR, ignore_errors=True)

    print_report(stats)

    logger.info(f"\n✅ Guardado: {OUTPUT_CSV}")
    logger.info("Próximo: python scripts/compute_embeddings.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
