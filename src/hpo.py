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


# ============================================================================
# EVALUACIÓN CON TORCH (requiere GPU; se verifica con el smoke run en Colab)
# ============================================================================

def evaluate_config(
    hp: dict,
    data: dict,
    inner_folds,
    device: str,
    epochs: int = 40,
    patience: int = 8,
    trial=None,
    seed: int = 42,
) -> list[float]:
    """
    Entrena una configuración en cada fold interno y devuelve su val_loss.

    El escalado se ajusta con el train de CADA fold interno, nunca con el
    conjunto completo: es el punto donde es más fácil filtrar información.

    Si se pasa `trial`, se reporta el resultado tras cada fold para que el
    pruner pueda abandonar configuraciones malas sin completar los tres — es lo
    que hace viable el esquema anidado en tiempo de cómputo.
    """
    import os
    import tempfile

    import torch

    from models import HybridPredictiveModel
    from train import Trainer, make_dataloader
    from utils import scale_price_sequences

    losses: list[float] = []

    for i, fold in enumerate(inner_folds):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)

        seqs_scaled, _ = scale_price_sequences(data["price_seqs"], fold.train_idx)

        model = HybridPredictiveModel(
            price_input_size=data["price_seqs"].shape[2],
            sentiment_dim=data["sentiments"].shape[1],
            hidden_size=hp["hidden_size"],
            d_model=hp["d_model"],
            num_heads=hp["num_heads"],
            num_lstm_layers=hp["num_lstm_layers"],
            dropout=hp["dropout"],
        )
        trainer = Trainer(
            model, device=device,
            lr=hp["lr"], weight_decay=hp["weight_decay"],
            w_t1=hp["w_t1"], w_t5=hp["w_t5"],
        )

        train_loader = make_dataloader(
            seqs_scaled, data["sentiments"], data["y_t1"], data["y_t5"],
            fold.train_idx, batch_size=hp["batch_size"], shuffle=True,
        )
        val_loader = make_dataloader(
            seqs_scaled, data["sentiments"], data["y_t1"], data["y_t5"],
            fold.val_idx, batch_size=hp["batch_size"],
        )

        # Checkpoint temporal: Trainer.fit lo necesita para restaurar los mejores
        # pesos. Se borra siempre — con cientos de trials, dejarlos acumularía
        # basura en el disco de Colab.
        fd, ckpt = tempfile.mkstemp(suffix=".pth")
        os.close(fd)
        try:
            history = trainer.fit(
                train_loader, val_loader,
                epochs=epochs, patience=patience, save_path=ckpt,
            )
        finally:
            if os.path.exists(ckpt):
                os.unlink(ckpt)

        losses.append(float(min(history["val_loss"])))

        del model, trainer
        if device == "cuda":
            torch.cuda.empty_cache()

        if trial is not None:
            trial.report(float(np.mean(losses)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return losses


def estimate_cost(data: dict, outer_folds, device: str, n_trials: int = 40) -> dict:
    """
    Cronometra UN trial real con la configuración por defecto y extrapola.

    El spec estima ~15h sin medir; esta función da el número real antes de
    comprometer horas de GPU.
    """
    import time

    hp = complete_hparams({
        "lr": 1e-3, "weight_decay": 1e-5, "hidden_size": 128,
        "d_model": 64, "dropout": 0.2, "w_t1": 0.6,
    })

    t0 = time.time()
    evaluate_config(hp, data, outer_folds[0].inner, device)
    seg_por_trial = time.time() - t0

    n_busquedas = len(outer_folds) + 1          # anidado + búsqueda final
    total_seg = seg_por_trial * n_trials * n_busquedas

    return {
        "segundos_por_trial": seg_por_trial,
        "trials_por_busqueda": n_trials,
        "busquedas": n_busquedas,
        "horas_totales_sin_pruning": total_seg / 3600,
        "horas_estimadas_con_pruning": total_seg / 3600 * 0.4,
    }
