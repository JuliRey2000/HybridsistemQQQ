"""
Descarga el dataset FNSPID desde Kaggle y lo normaliza al esquema estándar.

FNSPID (Financial News and Stock Price Integration Dataset) cubre noticias
financieras de 2009 a 2023 y es la fuente principal del corpus.

Requisitos:
  - CLI de Kaggle configurado: ~/.kaggle/kaggle.json
  - pip install kaggle

Output: data/raw/fnspid_news.csv
Schema: [date, headline, body]

Uso: python scripts/download_fnspid.py
"""

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW_PATH, START_DATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Kaggle dataset ─────────────────────────────────────────────────────────────
# Verifica el ID exacto en: https://www.kaggle.com/datasets
# Alternativa conocida: "omermetinn/financial-news-data-set"
KAGGLE_DATASET = "humananalog/fnspid"
OUTPUT_CSV     = DATA_RAW_PATH / "fnspid_news.csv"


# ── Columnas posibles por versión de FNSPID ────────────────────────────────────
DATE_COLS     = {"date", "publish_date", "publishdate", "published_at", "publisheddate", "time"}
HEADLINE_COLS = {"headline", "title", "article_title", "head", "subject"}
BODY_COLS     = {"body", "article", "content", "text", "article_text", "story", "description"}


def _find_col(df_cols: list[str], candidates: set[str]) -> str | None:
    for col in df_cols:
        if col.lower().strip() in candidates:
            return col
    return None


def download_from_kaggle() -> None:
    """Descarga el dataset vía CLI de Kaggle (unzip automático)."""
    logger.info(f"Descargando FNSPID desde Kaggle: {KAGGLE_DATASET}")
    logger.info("(Esto puede tardar varios minutos según el tamaño del dataset)")

    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(DATA_RAW_PATH),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Error al descargar desde Kaggle:\n{result.stderr}\n\n"
            "Posibles causas:\n"
            "  1. ~/.kaggle/kaggle.json no está configurado\n"
            "  2. El ID del dataset es incorrecto\n"
            "  3. No aceptaste los términos del dataset en kaggle.com\n"
            f"Verifica el ID en: https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
        )

    logger.info("Descarga completada.")


def find_fnspid_csv() -> Path:
    """Localiza el CSV principal de FNSPID entre los archivos descargados."""
    all_csvs = sorted(DATA_RAW_PATH.glob("*.csv"), key=lambda f: f.stat().st_size, reverse=True)

    if not all_csvs:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV en {DATA_RAW_PATH}.\n"
            "Verifica que la descarga de Kaggle completó correctamente."
        )

    # Preferir archivos con nombres relacionados con noticias financieras
    priority_keywords = ["fnspid", "financial", "news", "article"]
    for csv in all_csvs:
        if any(kw in csv.stem.lower() for kw in priority_keywords):
            logger.info(f"CSV seleccionado: {csv.name}  ({csv.stat().st_size / 1e6:.1f} MB)")
            return csv

    # Fallback: el CSV más grande (normalmente el principal)
    logger.info(f"CSV seleccionado (por tamaño): {all_csvs[0].name}  ({all_csvs[0].stat().st_size / 1e6:.1f} MB)")
    return all_csvs[0]


def normalize(raw_path: Path) -> pd.DataFrame:
    """
    Normaliza el CSV de FNSPID al esquema estándar: [date, headline, body].
    Maneja variantes de nombres de columna entre versiones del dataset.
    """
    logger.info(f"Leyendo {raw_path.name} ...")
    df = pd.read_csv(raw_path, low_memory=False)
    logger.info(f"Columnas encontradas: {list(df.columns)}")
    logger.info(f"Registros crudos    : {len(df):,}")

    col_date     = _find_col(list(df.columns), DATE_COLS)
    col_headline = _find_col(list(df.columns), HEADLINE_COLS)
    col_body     = _find_col(list(df.columns), BODY_COLS)

    if col_date is None or col_headline is None:
        raise ValueError(
            f"No se pudo identificar columnas obligatorias.\n"
            f"Columnas disponibles: {list(df.columns)}\n"
            f"Esperadas: date/headline en {DATE_COLS} | {HEADLINE_COLS}"
        )

    logger.info(f"Mapeando — date: '{col_date}', headline: '{col_headline}', body: '{col_body or 'N/A'}'")

    df = df.rename(columns={col_date: "date", col_headline: "headline"})
    if col_body:
        df = df.rename(columns={col_body: "body"})
    else:
        df["body"] = ""

    df = df[["date", "headline", "body"]].copy()

    # Parsear fechas — maneja múltiples formatos
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce", utc=True)
        .dt.normalize()
        .dt.tz_localize(None)
    )
    df = df.dropna(subset=["date", "headline"])

    df["headline"] = df["headline"].astype(str).str.strip()
    df["body"]     = df["body"].fillna("").astype(str).str.strip()

    # Eliminar headlines vacíos
    df = df[df["headline"] != ""]

    # Filtrar al rango de la tesis: START_DATE → 2023-12-31
    df = df[
        (df["date"] >= pd.Timestamp(START_DATE)) &
        (df["date"] <= pd.Timestamp("2023-12-31"))
    ]
    df = df.sort_values("date").reset_index(drop=True)

    return df


def print_report(df: pd.DataFrame) -> None:
    total = len(df)
    body_empty_pct = (df["body"] == "").mean() * 100
    days_with_news = df["date"].nunique()

    by_year = df.set_index("date").resample("YE").size()

    logger.info(f"\n{'='*60}")
    logger.info("REPORTE FNSPID")
    logger.info(f"{'='*60}")
    logger.info(f"  Total noticias     : {total:,}")
    logger.info(f"  Días únicos        : {days_with_news:,}")
    logger.info(f"  Rango de fechas    : {df['date'].min().date()} → {df['date'].max().date()}")
    logger.info(f"  Body vacío (%)     : {body_empty_pct:.1f}%  (normal si FNSPID no incluye cuerpo)")
    logger.info(f"  Noticias por año:")
    for year, count in by_year.items():
        logger.info(f"    {year.year}: {count:7,}")
    logger.info(f"{'='*60}")


def main() -> int:
    if OUTPUT_CSV.exists():
        logger.info(f"Archivo ya existe: {OUTPUT_CSV}")
        logger.info("Para re-descargar, elimina el archivo y vuelve a ejecutar.")
        return 0

    DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)

    download_from_kaggle()
    raw_path = find_fnspid_csv()
    df = normalize(raw_path)

    df.to_csv(OUTPUT_CSV, index=False)
    print_report(df)

    logger.info(f"\n✅ Guardado: {OUTPUT_CSV}")
    logger.info("Próximo: python scripts/download_tiingo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
