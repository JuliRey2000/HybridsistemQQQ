"""
Descarga noticias financieras de 2024 desde la API de Tiingo.

Tiingo cubre 2024 que no está en FNSPID (2009-2023).
Rate limit: ~10,000 requests/día en plan Basic (~$10/mes).
Este script hace ~12 requests mensuales + paginación — bien dentro del límite.

Requisitos:
  - Variable de entorno TIINGO_API_KEY
  - pip install requests

Obtén tu token gratuito/básico en: https://api.tiingo.com/
  Plan Basic ($10/mes): suficiente para 1 mes de descarga

Output: data/raw/tiingo_2024.csv
Schema: [date, headline, body]

Uso:
  export TIINGO_API_KEY=tu_token
  python scripts/download_tiingo.py
"""

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_CSV = DATA_RAW_PATH / "tiingo_2024.csv"
BASE_URL   = "https://api.tiingo.com/tiingo/news"

# Ventanas mensuales de 2024
MONTHS_2024 = [
    ("2024-01-01", "2024-01-31"),
    ("2024-02-01", "2024-02-29"),
    ("2024-03-01", "2024-03-31"),
    ("2024-04-01", "2024-04-30"),
    ("2024-05-01", "2024-05-31"),
    ("2024-06-01", "2024-06-30"),
    ("2024-07-01", "2024-07-31"),
    ("2024-08-01", "2024-08-31"),
    ("2024-09-01", "2024-09-30"),
    ("2024-10-01", "2024-10-31"),
    ("2024-11-01", "2024-11-30"),
    ("2024-12-01", "2024-12-31"),
]

PAGE_LIMIT     = 1000   # máx por request (límite de la API)
PAUSE_BETWEEN  = 0.5    # segundos entre requests (rate limiting)
PAUSE_MONTHLY  = 2.0    # pausa entre meses


def fetch_month(api_key: str, start: str, end: str) -> list[dict]:
    """
    Descarga noticias de un mes completo con paginación automática.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {api_key}",
    }
    params = {
        "startDate": start,
        "endDate":   end,
        "limit":     PAGE_LIMIT,
        "offset":    0,
    }

    records = []
    page    = 0

    while True:
        try:
            resp = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError as e:
            if resp.status_code == 401:
                raise EnvironmentError(
                    "API key inválida (401 Unauthorized).\n"
                    "Verifica TIINGO_API_KEY en: https://api.tiingo.com/account/api/token"
                ) from e
            elif resp.status_code == 429:
                logger.warning(f"Rate limit alcanzado. Esperando 60s...")
                time.sleep(60)
                continue
            raise

        batch = resp.json()
        if not batch:
            break

        records.extend(batch)
        page += 1

        logger.debug(f"  Página {page}: {len(batch)} noticias (offset={params['offset']})")

        if len(batch) < PAGE_LIMIT:
            break

        params["offset"] += PAGE_LIMIT
        time.sleep(PAUSE_BETWEEN)

    return records


def normalize_records(records: list[dict]) -> pd.DataFrame:
    """
    Normaliza la respuesta JSON de Tiingo al esquema estándar.

    Campos Tiingo: publishedDate, title, description, url, source, tags, tickers
    """
    rows = []
    for r in records:
        raw_date = r.get("publishedDate") or r.get("publishDate") or ""
        date = pd.to_datetime(raw_date, errors="coerce", utc=True)
        if pd.isna(date):
            continue

        headline = str(r.get("title", "")).strip()
        body     = str(r.get("description", "")).strip()

        if not headline:
            continue

        rows.append({
            "date":     date.normalize().tz_localize(None),
            "headline": headline,
            "body":     body,
        })

    if not rows:
        return pd.DataFrame(columns=["date", "headline", "body"])

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def print_report(df: pd.DataFrame) -> None:
    logger.info(f"\n{'='*60}")
    logger.info("REPORTE TIINGO 2024")
    logger.info(f"{'='*60}")
    logger.info(f"  Total noticias     : {len(df):,}")
    logger.info(f"  Días únicos        : {df['date'].nunique():,}")

    if len(df) > 0:
        by_month = df.set_index("date").resample("ME").size()
        logger.info("  Noticias por mes:")
        for month, count in by_month.items():
            marker = " ⚠️ VACÍO" if count == 0 else ""
            logger.info(f"    {month.strftime('%Y-%m')}: {count:5,}{marker}")

    logger.info(f"{'='*60}")


def main() -> int:
    if OUTPUT_CSV.exists():
        logger.info(f"Archivo ya existe: {OUTPUT_CSV}")
        logger.info("Para re-descargar, elimina el archivo y vuelve a ejecutar.")
        return 0

    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        logger.error(
            "Variable de entorno TIINGO_API_KEY no encontrada.\n"
            "Configura con:\n"
            "  export TIINGO_API_KEY=tu_token_aqui\n"
            "\nObtén tu token en: https://api.tiingo.com/account/api/token\n"
            "Plan Basic ($10/mes o gratis con límite 50req/hour) es suficiente."
        )
        return 1

    DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)

    all_records = []

    for start, end in MONTHS_2024:
        logger.info(f"Descargando {start} → {end} ...")
        try:
            records = fetch_month(api_key, start, end)
            all_records.extend(records)
            logger.info(f"  ✓ {len(records):5,} noticias")
        except Exception as e:
            logger.error(f"  Error en {start}: {e}")
        time.sleep(PAUSE_MONTHLY)

    if not all_records:
        logger.error("No se descargaron noticias. Verifica la API key y los parámetros.")
        return 1

    df = normalize_records(all_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print_report(df)

    logger.info(f"\n✅ Guardado: {OUTPUT_CSV}")
    logger.info("Próximo: python scripts/build_corpus.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
