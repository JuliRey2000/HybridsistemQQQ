"""
Tests de la maquinaria del estudio Optuna.

La función de evaluación se inyecta, así que estos tests corren sin torch ni
GPU: se comprueba la gestión del estudio, la reanudabilidad y el tratamiento de
trials fallidos, no el entrenamiento.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from hpo import records_from_study, run_search


def evaluador_sintetico(hp, trial):
    """val_loss determinista: óptimo en dropout bajo y hidden_size pequeño."""
    base = 1.0 + hp["dropout"] + hp["hidden_size"] / 1000.0
    return [base - 0.01, base, base + 0.01]


def test_corre_el_numero_de_trials_pedido(tmp_path):
    study = run_search(
        evaluador_sintetico, n_trials=6,
        study_name="t1", storage=f"sqlite:///{tmp_path}/a.db",
    )
    assert len(study.trials) == 6


def test_guarda_los_fold_losses_de_cada_trial(tmp_path):
    study = run_search(
        evaluador_sintetico, n_trials=4,
        study_name="t2", storage=f"sqlite:///{tmp_path}/b.db",
    )
    records = records_from_study(study)
    assert len(records) == 4
    for r in records:
        assert len(r.fold_losses) == 3
        assert np.isfinite(r.mean_loss)


def test_reanuda_sin_repetir_trials(tmp_path):
    """Una desconexión de Colab no debe costar el cómputo ya hecho."""
    storage = f"sqlite:///{tmp_path}/c.db"
    primero = run_search(evaluador_sintetico, n_trials=3,
                         study_name="reanuda", storage=storage)
    assert len(primero.trials) == 3

    segundo = run_search(evaluador_sintetico, n_trials=3,
                         study_name="reanuda", storage=storage)
    assert len(segundo.trials) == 6


def test_un_trial_que_revienta_no_tumba_el_estudio(tmp_path):
    llamadas = {"n": 0}

    def evaluador_inestable(hp, trial):
        llamadas["n"] += 1
        if llamadas["n"] == 2:
            raise RuntimeError("CUDA out of memory (simulado)")
        return evaluador_sintetico(hp, trial)

    study = run_search(evaluador_inestable, n_trials=4,
                       study_name="t3", storage=f"sqlite:///{tmp_path}/d.db")

    assert len(study.trials) == 4
    medias = [r.mean_loss for r in records_from_study(study)]
    assert sum(1 for m in medias if not np.isfinite(m)) == 1


def test_el_estudio_es_reproducible_con_la_misma_semilla(tmp_path):
    a = run_search(evaluador_sintetico, n_trials=5, seed=7,
                   study_name="s", storage=f"sqlite:///{tmp_path}/e.db")
    b = run_search(evaluador_sintetico, n_trials=5, seed=7,
                   study_name="s", storage=f"sqlite:///{tmp_path}/f.db")
    assert [t.params for t in a.trials] == [t.params for t in b.trials]
