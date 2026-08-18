"""
Maquinaria de la búsqueda de hiperparámetros.

Gestiona el estudio de Optuna, entrena y evalúa configuraciones con torch, y
orquesta el walk-forward anidado. La lógica que DECIDE (splits, espacio,
selección) vive en `hpo_core.py` y se verifica por separado.

`run_search` recibe la función de evaluación como parámetro: así la gestión del
estudio y la reanudabilidad son verificables sin GPU.
"""
from __future__ import annotations

import logging

import numpy as np
import optuna

from hpo_core import TrialRecord, complete_hparams, suggest_hparams, validate_hparams

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

FOLD_LOSSES_ATTR = "fold_losses"


def run_search(
    evaluate_fn,
    n_trials: int,
    study_name: str,
    storage: str,
    seed: int = 42,
    n_startup_trials: int = 10,
) -> optuna.Study:
    """
    Crea o reanuda un estudio y ejecuta `n_trials` evaluaciones.

    Args:
        evaluate_fn: `(hp: dict, trial) -> list[float]`, el val_loss de cada
                     fold interno para esa configuración.
        storage    : URL de SQLite. Persistir el estudio es lo que permite
                     reanudar tras una desconexión de Colab.

    Un trial que revienta (OOM, configuración inválida) devuelve infinito y
    queda registrado, en vez de tumbar la búsqueda entera.
    """
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_hparams(trial)
        try:
            validate_hparams(hp)
            fold_losses = list(evaluate_fn(hp, trial))
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.warning(f"Trial {trial.number} falló: {type(exc).__name__}: {exc}")
            trial.set_user_attr(FOLD_LOSSES_ATTR, [float("inf")])
            return float("inf")

        trial.set_user_attr(FOLD_LOSSES_ATTR, fold_losses)
        return float(np.mean(fold_losses))

    study.optimize(objective, n_trials=n_trials)
    return study


def records_from_study(study: optuna.Study) -> list[TrialRecord]:
    """Convierte los trials completados en TrialRecord para la regla de 1-SE."""
    records = []
    for trial in study.trials:
        losses = trial.user_attrs.get(FOLD_LOSSES_ATTR)
        if not losses:
            continue
        records.append(
            TrialRecord(
                params=complete_hparams(dict(trial.params)),
                fold_losses=tuple(float(x) for x in losses),
            )
        )
    return records
