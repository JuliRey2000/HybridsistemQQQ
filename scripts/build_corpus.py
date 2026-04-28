"""
Unifica y limpia el corpus de noticias (FNSPID + Tiingo 2024).

Pasos:
  1. Carga fnspid_news.csv + tiingo_2024.csv
  2. Concatena y elimina duplicados por (date, headline)
  3. Ordena cronológicamente
  4. Guarda data/interim/corpus_merged.csv

Output: data/interim/corpus_merged.csv
Schema: [date, headline, body]

Uso: python scripts/build_corpus.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

INTERIM_DIR  = Path(__file__).parent.parent / "data" / "interim"
FNSPID_CSV   = DATA_RAW_PATH / "fnspid_news.csv"
TIINGO_CSV   = DATA_RAW_PATH / "tiingo_2024.csv"
OUTPUT_CSV   = INTERIM_DIR / "corpus_merged.csv"


def load_source(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"{label} no encontrado en {path}. Omitiendo esta fuente.")
        return pd.DataFrame(columns=["date", "headline", "body"])

    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "headline"])
    df["headline"] = df["headline"].astype(str).str.strip()
    df["body"]     = df["body"].fillna("").astype(str).str.strip()
    df = df[df["headline"] != ""]

    logger.info(
        f"{label:12s}: {len(df):7,} noticias  "
        f"[{df['date'].min().date()} → {df['date'].max().date()}]  "
        f"body_empty={( df['body'] == '' ).mean()*100:.0f}%"
    )
    return df[["date", "headline", "body"]].copy()


def main() -> int:
    if OUTPUT_CSV.exists():
        logger.info(f"Corpus ya existe: {OUTPUT_CSV}")
        logger.info("Para reconstruir, elimina el archivo y vuelve a ejecutar.")
        return 0

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    fnspid = load_source(FNSPID_CSV, "FNSPID")
    tiingo = load_source(TIINGO_CSV, "Tiingo")

    if fnspid.empty and tiingo.empty:
        logger.error(
            "No se encontraron fuentes de noticias.\n"
            "Ejecuta primero:\n"
            "  python scripts/download_fnspid.py\n"
            "  python scripts/download_tiingo.py"
        )
        return 1

    combined = pd.concat([fnspid, tiingo], ignore_index=True)
    n_before = len(combined)

    # Deduplicar por (date, headline) — el mismo título el mismo día es duplicado
    combined = combined.drop_duplicates(subset=["date", "headline"], keep="first")
    n_dupes  = n_before - len(combined)

    combined = combined.sort_values("date").reset_index(drop=True)

    # Cobertura: % de días con ≥1 noticia en el rango 2015-2024
    all_market_range = pd.date_range(start="2015-01-01", end="2024-12-31", freq="B")  # días hábiles aprox
    covered_days     = combined["date"].nunique()
    coverage_pct     = covered_days / len(all_market_range) * 100

    combined.to_csv(OUTPUT_CSV, index=False)

    logger.info(f"\n{'='*60}")
    logger.info("REPORTE CORPUS MERGED")
    logger.info(f"{'='*60}")
    logger.info(f"  Total noticias         : {len(combined):,}")
    logger.info(f"  Duplicados eliminados  : {n_dupes:,}")
    logger.info(f"  Días únicos con noticias: {covered_days:,}")
    logger.info(f"  Cobertura (aprox)      : {coverage_pct:.1f}%  de días hábiles 2015-2024")
    logger.info(f"  Rango final            : {combined['date'].min().date()} → {combined['date'].max().date()}")

    by_year = combined.set_index("date").resample("YE").size()
    logger.info("  Noticias por año:")
    for year, count in by_year.items():
        logger.info(f"    {year.year}: {count:7,}")

    logger.info(f"{'='*60}")
    logger.info(f"\n✅ Guardado: {OUTPUT_CSV}")
    logger.info("Próximo: python scripts/compute_embeddings.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
