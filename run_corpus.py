"""
Orquestador del corpus de noticias + embeddings FinBERT.

Ejecuta la cadena completa en orden:
  1. download_fnspid.py   → data/raw/fnspid_news.csv
  2. download_tiingo.py   → data/raw/tiingo_2024.csv
  3. build_corpus.py      → data/interim/corpus_merged.csv
  4. compute_embeddings.py → data/processed/finbert_embeddings.csv

Cada script es idempotente: si su output ya existe, lo omite.
Puedes re-ejecutar run_corpus.py sin riesgo de sobreescribir trabajo previo.

Requisitos previos:
  - ~/.kaggle/kaggle.json configurado  (para FNSPID)
  - Variable de entorno TIINGO_API_KEY  (para Tiingo 2024)
  - python run_pipeline.py ejecutado    (genera price_df.csv, necesario para compute_embeddings)
  - pip install transformers torch tqdm kaggle requests

Uso:
  export TIINGO_API_KEY=tu_token
  python run_corpus.py
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent / "scripts"

STEPS = [
    {
        "name"  : "FNSPID download",
        "script": SCRIPTS_DIR / "download_fnspid.py",
        "output": Path(__file__).parent / "data" / "raw" / "fnspid_news.csv",
    },
    {
        "name"  : "Tiingo 2024 download",
        "script": SCRIPTS_DIR / "download_tiingo.py",
        "output": Path(__file__).parent / "data" / "raw" / "tiingo_2024.csv",
    },
    {
        "name"  : "Build corpus merged",
        "script": SCRIPTS_DIR / "build_corpus.py",
        "output": Path(__file__).parent / "data" / "interim" / "corpus_merged.csv",
    },
    {
        "name"  : "Compute FinBERT embeddings",
        "script": SCRIPTS_DIR / "compute_embeddings.py",
        "output": Path(__file__).parent / "data" / "processed" / "finbert_embeddings.csv",
    },
]


def run_step(step: dict) -> bool:
    """Ejecuta un paso. Retorna True si tuvo éxito."""
    name   = step["name"]
    script = step["script"]
    output = step["output"]

    if output.exists():
        logger.info(f"[SKIP] {name} — output ya existe: {output.name}")
        return True

    logger.info(f"\n{'='*65}")
    logger.info(f"[RUN]  {name}")
    logger.info(f"{'='*65}")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(Path(__file__).parent),
    )

    if result.returncode != 0:
        logger.error(f"[FAIL] {name} salió con código {result.returncode}")
        return False

    if not output.exists():
        logger.error(f"[FAIL] {name} finalizó pero el output no existe: {output}")
        return False

    logger.info(f"[OK]   {name} → {output.name}")
    return True


def main() -> int:
    logger.info("CORPUS PIPELINE — FinBERT embeddings para QQQ")
    logger.info(f"  Pasos: {len(STEPS)}")
    logger.info(f"  Scripts: {SCRIPTS_DIR}")

    for i, step in enumerate(STEPS, start=1):
        logger.info(f"\nPaso {i}/{len(STEPS)}: {step['name']}")
        ok = run_step(step)
        if not ok:
            logger.error(f"\nPipeline detenido en paso {i}: {step['name']}")
            logger.error("Revisa el error arriba y vuelve a ejecutar.")
            logger.error("Los pasos completados NO se re-ejecutarán (idempotentes).")
            return 1

    logger.info(f"\n{'='*65}")
    logger.info("CORPUS PIPELINE COMPLETADO")
    logger.info(f"{'='*65}")
    logger.info("  ✅ fnspid_news.csv")
    logger.info("  ✅ tiingo_2024.csv")
    logger.info("  ✅ corpus_merged.csv")
    logger.info("  ✅ finbert_embeddings.csv")
    logger.info("\nPróximo:")
    logger.info("  python run_pipeline.py          (regenera price_seqs.npy con sentimiento real)")
    logger.info("  python run_train_predictive.py  (entrena HybridPredictiveModel)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
