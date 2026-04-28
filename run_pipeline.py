"""
Script de ejecución: FASE 1 — Descarga y preprocesamiento de datos.

Ejecutar: python run_pipeline.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import DataPipeline
from config import (
    TICKER, START_DATE, END_DATE, LOOKBACK,
    SENTIMENT_PRECOMP, NEWS_CSV_PATH,
    DATA_PROCESSED_PATH, LOG_LEVEL, print_config,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def main() -> int:
    print_config()
    logger.info("FASE 1 — DATA PIPELINE")

    pipeline = DataPipeline(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback=LOOKBACK,
        sentiment_path=SENTIMENT_PRECOMP if Path(SENTIMENT_PRECOMP).exists() else None,
        news_csv_path=NEWS_CSV_PATH if Path(NEWS_CSV_PATH).exists() else None,
        save_dir=str(DATA_PROCESSED_PATH),
    )

    data = pipeline.run()

    print("\n✅ PIPELINE COMPLETADO")
    print(f"   price_seqs : {data['price_seqs'].shape}")
    print(f"   sentiments : {data['sentiments'].shape}")
    print(f"   y_t1       : {data['y_t1'].shape}")
    print(f"   y_t5       : {data['y_t5'].shape}")
    print(f"\nPróximo: python run_train_predictive.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
