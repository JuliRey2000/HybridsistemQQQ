"""
Descarga el dataset FNSPID desde Hugging Face y lo normaliza al esquema estándar.

FNSPID (Financial News and Stock Price Integration Dataset) es la fuente
principal del corpus de noticias 2015-2023.

Fuente oficial de los autores, sin login ni token:
  https://huggingface.co/datasets/Zihan1004/FNSPID

Se usa `Stock_news/nasdaq_exteral_data.csv` (23.2 GB) leyendo ÚNICAMENTE el
titular, nunca el cuerpo del artículo.

Por qué este archivo y no `All_external.csv` (5.7 GB): se probó primero el
segundo por ser cuatro veces menor, pero **solo llega hasta 2020-06-11**. Eso
dejaba un hueco de tres años y medio (2020-06 a 2023-12) justo antes de la
ventana de test. El archivo grande cubre de 2003 a 2023-12 (verificado
muestreando siete puntos del archivo).

Por qué solo el titular: Tiingo — la fuente de 2024 — entrega titular +
descripción corta. Mezclar artículos completos en train con textos cortos en
test haría que el embedding de FinBERT cambie de naturaleza dentro de la ventana
de evaluación. Además el cuerpo se truncaría a 512 tokens, subiendo el cómputo
de FinBERT de ~3h a varios días.

Overrides por variable de entorno: FNSPID_URL para cambiar de archivo,
FNSPID_USE_BODY=1 para incorporar el cuerpo (cambia la metodología, ver arriba).

El archivo es demasiado grande para pd.read_csv() en Colab (12.7 GB de RAM),
así que se descarga en streaming y se filtra por chunks: el CSV crudo nunca se
materializa ni en disco ni en memoria.

Requisitos:
  - pip install requests pandas tqdm     (sin credenciales)

Output: data/raw/fnspid_news.csv
Schema: [date, headline, body]

Uso: python scripts/download_fnspid.py
"""

import logging
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW_PATH, START_DATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Fuente ─────────────────────────────────────────────────────────────────────
HF_REPO     = "Zihan1004/FNSPID"
HF_FILE     = "Stock_news/nasdaq_exteral_data.csv"   # 23.2 GB, cobertura 2003-2023
DEFAULT_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_FILE}"

FNSPID_URL  = os.getenv("FNSPID_URL", DEFAULT_URL)
OUTPUT_CSV  = DATA_RAW_PATH / "fnspid_news.csv"

# Solo titulares. Incorporar el cuerpo cambia la metodología (ver docstring) y
# multiplica por decenas el cómputo de FinBERT, así que es opt-in explícito.
USE_BODY    = os.getenv("FNSPID_USE_BODY", "0") == "1"

# FNSPID llega hasta 2023; 2024 lo cubre Tiingo (ver download_tiingo.py)
FNSPID_END  = "2023-12-31"

CHUNKSIZE   = 500_000    # filas por chunk — ~200 MB de RAM pico con 3 columnas


# ── Columnas posibles por versión de FNSPID ────────────────────────────────────
DATE_COLS     = {"date", "publish_date", "publishdate", "published_at", "publisheddate", "time"}
HEADLINE_COLS = {"headline", "title", "article_title", "head", "subject"}
BODY_COLS     = {"body", "article", "content", "text", "article_text", "story", "description"}


def _find_col(df_cols: list[str], candidates: set[str]) -> str | None:
    for col in df_cols:
        if col.lower().strip() in candidates:
            return col
    return None


class _ProgressReader:
    """Envuelve un stream binario para alimentar la barra de progreso al leerlo."""

    def __init__(self, raw, bar: tqdm):
        self._raw = raw
        self._bar = bar

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)
        self._bar.update(len(chunk))
        return chunk


def fetch_header(url: str) -> list[str]:
    """Lee solo los primeros bytes del CSV remoto para conocer sus columnas."""
    logger.info(f"Leyendo cabecera de {HF_FILE} ...")
    resp = requests.get(url, headers={"Range": "bytes=0-8191"}, timeout=60)
    resp.raise_for_status()

    first_line = resp.text.splitlines()[0]
    cols = [c.strip().strip('"') for c in first_line.split(",")]
    logger.info(f"Columnas encontradas: {cols}")
    return cols


def resolve_columns(cols: list[str]) -> tuple[str, str, str | None]:
    """Mapea las columnas del CSV al esquema [date, headline, body]."""
    col_date     = _find_col(cols, DATE_COLS)
    col_headline = _find_col(cols, HEADLINE_COLS)
    col_body     = _find_col(cols, BODY_COLS) if USE_BODY else None

    if col_date is None or col_headline is None:
        raise ValueError(
            f"No se pudo identificar columnas obligatorias.\n"
            f"Columnas disponibles: {cols}\n"
            f"Esperadas: date/headline en {DATE_COLS} | {HEADLINE_COLS}"
        )

    logger.info(
        f"Mapeando — date: '{col_date}', headline: '{col_headline}', "
        f"body: '{col_body or 'omitido (solo titulares)'}'"
    )
    return col_date, col_headline, col_body


def parse_dates(raw: pd.Series) -> pd.Series:
    """
    Parsea fechas tipo '2020-06-05 06:30:54 UTC'.

    Ruta rápida: los primeros 19 caracteres con formato explícito (todo FNSPID
    viene en UTC). Las filas que fallen se reintentan con el parser flexible,
    que es órdenes de magnitud más lento y no conviene aplicar a millones.
    """
    s = raw.astype(str).str.strip()
    parsed = pd.to_datetime(s.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")

    failed = parsed.isna() & s.ne("") & s.ne("nan")
    if failed.any():
        parsed.loc[failed] = pd.to_datetime(
            s.loc[failed], errors="coerce", utc=True
        ).dt.tz_localize(None)

    return parsed.dt.normalize()


def normalize_chunk(
    chunk: pd.DataFrame,
    col_date: str,
    col_headline: str,
    col_body: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Normaliza un chunk al esquema [date, headline, body] y lo filtra al rango."""
    df = chunk.rename(columns={col_date: "date", col_headline: "headline"})
    if col_body:
        df = df.rename(columns={col_body: "body"})
    else:
        df["body"] = ""

    df["date"] = parse_dates(df["date"])
    df = df.dropna(subset=["date", "headline"])

    # Filtrar el rango de la tesis antes de limpiar strings (mucho más barato)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if df.empty:
        return df[["date", "headline", "body"]]

    df["headline"] = df["headline"].astype(str).str.strip()
    df["body"]     = df["body"].fillna("").astype(str).str.strip()
    df = df[df["headline"] != ""]

    return df[["date", "headline", "body"]]


def stream_and_normalize(url: str, tmp_path: Path) -> dict:
    """
    Descarga el CSV en streaming, filtra chunk por chunk y escribe el resultado
    incrementalmente en `tmp_path`. Devuelve estadísticas para el reporte.

    Deduplica globalmente por (date, headline): FNSPID repite el mismo titular
    una vez por cada ticker mencionado, y como el sentimiento se agrega por día
    esas filas son duplicados exactos. Se guardan hashes, no los textos, para
    que el set quepa en RAM.
    """
    cols = fetch_header(url)
    col_date, col_headline, col_body = resolve_columns(cols)

    usecols = [c for c in (col_date, col_headline, col_body) if c]
    start   = pd.Timestamp(START_DATE)
    end     = pd.Timestamp(FNSPID_END)

    logger.info(f"Rango a conservar: {start.date()} → {end.date()}")
    logger.info("Descargando y filtrando en streaming (no se guarda el CSV crudo)...")

    seen: set[int] = set()
    stats = {
        "raw_rows": 0, "kept": 0, "dupes": 0,
        "by_year": Counter(), "days": set(),
        "min_date": None, "max_date": None, "body_nonempty": 0,
    }

    resp = requests.get(url, stream=True, timeout=(30, 300))
    resp.raise_for_status()
    resp.raw.decode_content = True

    total = int(resp.headers.get("Content-Length", 0)) or None
    first_write = True

    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
              desc="FNSPID") as bar:
        reader = pd.read_csv(
            _ProgressReader(resp.raw, bar),
            chunksize=CHUNKSIZE,
            usecols=usecols,
            low_memory=False,
            on_bad_lines="skip",
        )

        with open(tmp_path, "w", encoding="utf-8", newline="") as out:
            for chunk in reader:
                stats["raw_rows"] += len(chunk)

                df = normalize_chunk(chunk, col_date, col_headline, col_body, start, end)
                if df.empty:
                    continue

                # Dedupe global por (date, headline) vía hash. El archivo está
                # agrupado por ticker, así que el mismo titular reaparece a GB de
                # distancia: deduplicar por chunk no bastaría.
                keys = pd.util.hash_pandas_object(
                    df[["date", "headline"]], index=False
                ).to_numpy()

                mask = []
                for key in keys.tolist():
                    is_new = key not in seen
                    if is_new:
                        seen.add(key)
                    mask.append(is_new)

                stats["dupes"] += len(df) - sum(mask)
                df = df[mask]
                if df.empty:
                    continue

                stats["kept"] += len(df)
                stats["body_nonempty"] += int((df["body"] != "").sum())
                stats["by_year"].update(df["date"].dt.year.tolist())
                stats["days"].update(df["date"].tolist())

                chunk_min, chunk_max = df["date"].min(), df["date"].max()
                if stats["min_date"] is None:
                    stats["min_date"], stats["max_date"] = chunk_min, chunk_max
                else:
                    stats["min_date"] = min(stats["min_date"], chunk_min)
                    stats["max_date"] = max(stats["max_date"], chunk_max)

                df.to_csv(out, header=first_write, index=False)
                first_write = False

    if first_write:
        raise RuntimeError(
            "No se conservó ninguna fila. Revisa el rango de fechas "
            f"({start.date()} → {end.date()}) o el formato de la columna de fecha."
        )

    return stats


def print_report(stats: dict) -> None:
    kept = stats["kept"]
    body_empty_pct = (1 - stats["body_nonempty"] / kept) * 100 if kept else 100.0

    logger.info(f"\n{'='*60}")
    logger.info("REPORTE FNSPID")
    logger.info(f"{'='*60}")
    logger.info(f"  Filas crudas leídas : {stats['raw_rows']:,}")
    logger.info(f"  Duplicados quitados : {stats['dupes']:,}  (mismo titular en varios tickers)")
    logger.info(f"  Total noticias      : {kept:,}")
    logger.info(f"  Días únicos         : {len(stats['days']):,}")
    logger.info(f"  Rango de fechas     : {stats['min_date'].date()} → {stats['max_date'].date()}")
    logger.info(f"  Body vacío (%)      : {body_empty_pct:.1f}%  (100% es lo esperado: All_external es solo titulares)")
    logger.info("  Noticias por año:")
    for year in sorted(stats["by_year"]):
        logger.info(f"    {year}: {stats['by_year'][year]:8,}")
    logger.info(f"{'='*60}")

    # Guard de cobertura. El error más caro de este script es descubrir tarde que
    # al corpus le faltan años: con All_external.csv los datos se cortaban en
    # 2020-06 y eso solo se vio DESPUÉS de descargar 5.7 GB y mirar el reporte a
    # mano. Los días sin noticias acaban con sentimiento forward-filled, que es
    # justo lo que vacía de contenido a la rama de FinBERT.
    esperado = pd.Timestamp(FNSPID_END)
    meses_sin = (esperado - stats["max_date"]).days / 30.44
    if meses_sin > 3:
        logger.warning(
            f"\n⚠️  COBERTURA INCOMPLETA\n"
            f"    Los datos acaban en {stats['max_date'].date()} pero se esperaban "
            f"hasta {esperado.date()}.\n"
            f"    Faltan ~{meses_sin:.0f} meses, que quedarán con sentimiento "
            f"forward-filled.\n"
            f"    Revisa si el archivo de origen ({HF_FILE}) cubre el rango completo."
        )


def main() -> int:
    if OUTPUT_CSV.exists():
        logger.info(f"Archivo ya existe: {OUTPUT_CSV}")
        logger.info("Para re-descargar, elimina el archivo y vuelve a ejecutar.")
        return 0

    DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)

    # Escribir a un temporal y renombrar al final: una descarga interrumpida no
    # debe dejar un CSV parcial que main() confunda con uno completo.
    tmp_path = OUTPUT_CSV.with_suffix(".csv.part")

    try:
        stats = stream_and_normalize(FNSPID_URL, tmp_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        logger.error(
            "Falló la descarga desde Hugging Face.\n"
            f"URL: {FNSPID_URL}\n"
            "Posibles causas:\n"
            "  1. Sin conexión o corte a mitad de la transferencia (reintenta)\n"
            "  2. El repo cambió de ruta — verifica en:\n"
            f"     https://huggingface.co/datasets/{HF_REPO}/tree/main/Stock_news"
        )
        raise

    tmp_path.replace(OUTPUT_CSV)
    print_report(stats)

    logger.info(f"\n✅ Guardado: {OUTPUT_CSV}")
    logger.info("Próximo: python scripts/download_tiingo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
