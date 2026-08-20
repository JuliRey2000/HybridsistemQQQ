"""
Orquestador del corpus de noticias + embeddings FinBERT.

Ejecuta la cadena completa en orden:
  1. download_fnspid.py   → data/raw/fnspid_news.csv
  2. download_tiingo.py   → data/raw/tiingo_2024.csv   (solo si END_DATE > 2023)
  3. build_corpus.py      → data/interim/corpus_merged.csv
  4. compute_embeddings.py → data/processed/finbert_embeddings.csv

El paso 2 se omite automáticamente cuando el período de estudio acaba en 2023 o
antes, que es hasta donde llega FNSPID. Solo hace falta para extender el corpus
a 2024, y la API de noticias de Tiingo requiere plan de pago.

Cada script es idempotente: si su output ya existe, lo omite.
Puedes re-ejecutar run_corpus.py sin riesgo de sobreescribir trabajo previo.

Requisitos previos:
  - python run_pipeline.py ejecutado  (genera price_df.csv, necesario para compute_embeddings)
  - pip install transformers torch tqdm requests
  - TIINGO_API_KEY solo si END_DATE va más allá de 2023

Uso:
  python run_corpus.py
"""

import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import END_DATE

# Última fecha que cubre FNSPID. Más allá haría falta otra fuente.
FNSPID_COVERAGE_END = "2023-12-31"

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
        # Se ejecuta siempre, aunque el CSV ya exista: el script comprueba en
        # segundos que el archivo en disco cubra el período completo. Saltarlo
        # es lo que dejó pasar el FNSPID que solo llegaba a 2020-06.
        "always_run": True,
    },
    {
        "name"  : "Tiingo 2024 download",
        "script": SCRIPTS_DIR / "download_tiingo.py",
        "output": Path(__file__).parent / "data" / "raw" / "tiingo_2024.csv",
        # Solo necesario si el estudio va más allá de la cobertura de FNSPID
        "skip_if": lambda: END_DATE <= FNSPID_COVERAGE_END,
        "skip_msg": (
            f"END_DATE={END_DATE} está dentro de la cobertura de FNSPID "
            f"(hasta {FNSPID_COVERAGE_END}): no hace falta Tiingo."
        ),
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

    skip_if = step.get("skip_if")
    if skip_if is not None and skip_if():
        logger.info(f"[SKIP] {name} — {step.get('skip_msg', 'no aplica')}")
        return True

    # Un paso `always_run` es idempotente por su cuenta y además revalida lo que
    # encuentre en disco, así que no se salta nunca.
    if output.exists() and not step.get("always_run"):
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
    logger.info(f"  Período: hasta {END_DATE}")

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
    if END_DATE > FNSPID_COVERAGE_END:
        logger.info("  ✅ tiingo_2024.csv")
    logger.info("  ✅ corpus_merged.csv")
    logger.info("  ✅ finbert_embeddings.csv")
    logger.info("\nPróximo:")
    logger.info("  python run_pipeline.py          (regenera price_seqs.npy con sentimiento real)")
    logger.info("  python run_train_predictive.py  (entrena HybridPredictiveModel)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
