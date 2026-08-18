"""
Búsqueda de hiperparámetros del HybridPredictiveModel.

Modos:
  estimate  cronometra un trial real y extrapola el costo (correr SIEMPRE primero)
  nested    walk-forward anidado — la estimación insesgada del procedimiento
  final     búsqueda sobre todo el train+val -> results/best_hparams.json

Uso:
  python run_hpo.py --mode estimate
  python run_hpo.py --mode nested --trials 40
  python run_hpo.py --mode final  --trials 40

La búsqueda es reanudable: si Colab se desconecta, volver a lanzar el mismo
comando continúa desde donde iba (el estudio vive en results/hpo.db).
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DATA_PROCESSED_PATH, DEVICE, RESULTS_PATH, TEST_FRAC
from hpo_core import build_nested_splits
from utils import final_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARRAYS = ["price_seqs", "sentiments", "y_t1", "y_t5"]


def cargar_datos() -> dict:
    """Carga las secuencias ya preprocesadas por run_pipeline.py."""
    faltan = [f"{a}.npy" for a in ARRAYS
              if not (DATA_PROCESSED_PATH / f"{a}.npy").exists()]
    if faltan:
        raise FileNotFoundError(
            f"Faltan {faltan} en {DATA_PROCESSED_PATH}.\n"
            "Ejecuta primero: python run_pipeline.py"
        )
    return {a: np.load(DATA_PROCESSED_PATH / f"{a}.npy") for a in ARRAYS}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["estimate", "nested", "final"], required=True)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--inner-folds", type=int, default=3)
    args = ap.parse_args()

    import hpo

    try:
        data = cargar_datos()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    n = len(data["y_t1"])
    test_start, _ = final_test_split(n, TEST_FRAC)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{RESULTS_PATH / 'hpo.db'}"

    tiene_sentimiento = bool(np.any(data["sentiments"] != 0))

    logger.info(f"Muestras     : {n}")
    logger.info(f"test_start   : {test_start}  (test OOS: {n - test_start} muestras, intocable)")
    logger.info(f"Dispositivo  : {DEVICE}")
    logger.info(f"FinBERT      : {'REAL' if tiene_sentimiento else 'CEROS (sin corpus)'}")

    if not tiene_sentimiento:
        logger.warning(
            "La rama de sentimiento recibe ceros. Los óptimos de d_model, dropout "
            "y w_t1 cambiarán cuando exista el corpus FinBERT: considera esperar."
        )

    if args.mode == "estimate":
        folds = build_nested_splits(test_start, n_inner=args.inner_folds)
        est = hpo.estimate_cost(data, folds, DEVICE, n_trials=args.trials)
        print("\n" + "=" * 60)
        print("ESTIMACIÓN DE COSTO")
        print("=" * 60)
        print(f"  Segundos por trial          : {est['segundos_por_trial']:.1f}")
        print(f"  Trials por búsqueda         : {est['trials_por_busqueda']}")
        print(f"  Búsquedas (anidado + final) : {est['busquedas']}")
        print(f"  Horas sin pruning           : {est['horas_totales_sin_pruning']:.1f}")
        print(f"  Horas estimadas con pruning : {est['horas_estimadas_con_pruning']:.1f}")
        print("=" * 60)
        print("\nSi el número es aceptable:  python run_hpo.py --mode nested")
        return 0

    if args.mode == "nested":
        resumen = hpo.nested_walk_forward(
            data, test_start, args.trials, storage, DEVICE, RESULTS_PATH
        )
        print(f"\nval_loss externo medio: {resumen['outer_val_loss_media']:.6f} "
              f"± {resumen['outer_val_loss_std']:.6f}")
        print(f"Detalle por fold en: {RESULTS_PATH / 'hpo_nested.csv'}")
        print("\nSiguiente:  python run_hpo.py --mode final")
        return 0

    params = hpo.final_search(
        data, test_start, args.trials, storage, DEVICE, RESULTS_PATH
    )
    print("\nConfiguración final:")
    for k, v in sorted(params.items()):
        print(f"  {k:16s}: {v}")
    print(f"\nGuardada en: {RESULTS_PATH / 'best_hparams.json'}")
    print("Siguiente: abrir QQQ_Hibrido_Completo.ipynb y ejecutar las secciones 3-5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
