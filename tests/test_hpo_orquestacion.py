"""
Tests del cableado de la orquestación anidada.

`evaluate_config` se sustituye por un evaluador sintético, así que estos tests
corren sin torch ni GPU. No verifican el entrenamiento — verifican que el
protocolo anidado llama a las piezas correctas, con los índices correctos, y
escribe los artefactos que el documento necesita.

Es el test que atrapa un bug de cableado antes de descubrirlo a las tres horas
de una corrida en Colab.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import hpo
from hpo_core import build_nested_splits

TEST_START = 400          # dataset sintético pequeño para que los tests sean rápidos
N_TOTAL    = 470


@pytest.fixture
def datos_sinteticos():
    rng = np.random.default_rng(0)
    return {
        "price_seqs": rng.normal(size=(N_TOTAL, 30, 10)).astype(np.float32),
        "sentiments": np.zeros((N_TOTAL, 768), dtype=np.float32),
        "y_t1":       rng.normal(size=N_TOTAL).astype(np.float32),
        "y_t5":       rng.normal(size=N_TOTAL).astype(np.float32),
    }


@pytest.fixture
def evaluador_falso(monkeypatch):
    """Sustituye evaluate_config y registra con qué índices se le llamó."""
    llamadas = []

    def fake(hp, data, folds, device, epochs=40, patience=8, trial=None, seed=42):
        llamadas.append({
            "hp": dict(hp),
            "folds": [(f.train_idx.copy(), f.val_idx.copy()) for f in folds],
            "epochs": epochs,
            "patience": patience,
        })
        # val_loss determinista y dependiente de la config, para que la búsqueda
        # tenga un óptimo real que encontrar
        base = 1.0 + hp["dropout"] + hp["hidden_size"] / 1000.0
        return [base + 0.01 * i for i in range(len(folds))]

    monkeypatch.setattr(hpo, "evaluate_config", fake)
    return llamadas


def test_nested_nunca_evalua_con_indices_del_test(datos_sinteticos, evaluador_falso, tmp_path):
    """La garantía central, comprobada sobre las llamadas reales."""
    hpo.nested_walk_forward(
        datos_sinteticos, TEST_START, n_trials=3,
        storage=f"sqlite:///{tmp_path}/n.db", device="cpu", results_dir=tmp_path,
    )
    assert evaluador_falso, "no se llamó al evaluador"
    for llamada in evaluador_falso:
        for train_idx, val_idx in llamada["folds"]:
            assert train_idx.max() < TEST_START
            assert val_idx.max()   < TEST_START


def test_nested_reentrena_la_ganadora_con_el_presupuesto_completo(
    datos_sinteticos, evaluador_falso, tmp_path
):
    """Búsqueda con 40 épocas; la ganadora se reentrena con 100 y patience 15."""
    hpo.nested_walk_forward(
        datos_sinteticos, TEST_START, n_trials=3,
        storage=f"sqlite:///{tmp_path}/n2.db", device="cpu", results_dir=tmp_path,
    )
    completos = [c for c in evaluador_falso if c["epochs"] == 100]
    assert len(completos) == 5, "debe reentrenarse una vez por fold externo"
    for c in completos:
        assert c["patience"] == 15
        assert len(c["folds"]) == 1     # el fold externo, envuelto como uno solo


def test_el_reentrenamiento_usa_el_train_y_val_externos(
    datos_sinteticos, evaluador_falso, tmp_path
):
    hpo.nested_walk_forward(
        datos_sinteticos, TEST_START, n_trials=3,
        storage=f"sqlite:///{tmp_path}/n3.db", device="cpu", results_dir=tmp_path,
    )
    externos = build_nested_splits(TEST_START)
    completos = [c for c in evaluador_falso if c["epochs"] == 100]

    for outer, llamada in zip(externos, completos):
        train_idx, val_idx = llamada["folds"][0]
        assert np.array_equal(train_idx, outer.train_idx)
        assert np.array_equal(val_idx,   outer.val_idx)


def test_nested_escribe_los_artefactos_del_documento(
    datos_sinteticos, evaluador_falso, tmp_path
):
    resumen = hpo.nested_walk_forward(
        datos_sinteticos, TEST_START, n_trials=3,
        storage=f"sqlite:///{tmp_path}/n4.db", device="cpu", results_dir=tmp_path,
    )
    assert (tmp_path / "hpo_nested.csv").exists()
    assert (tmp_path / "hpo_nested_resumen.json").exists()
    assert len(resumen["folds"]) == 5
    assert np.isfinite(resumen["outer_val_loss_media"])


def test_final_search_produce_best_hparams_completo(
    datos_sinteticos, evaluador_falso, tmp_path
):
    params = hpo.final_search(
        datos_sinteticos, TEST_START, n_trials=4,
        storage=f"sqlite:///{tmp_path}/f.db", device="cpu", results_dir=tmp_path,
    )

    guardado = json.loads((tmp_path / "best_hparams.json").read_text(encoding="utf-8"))
    assert guardado == params

    # El notebook lee exactamente estas claves; si falta una, la celda revienta
    for clave in ["hidden_size", "d_model", "num_heads", "num_lstm_layers",
                  "dropout", "lr", "weight_decay", "w_t1", "w_t5", "batch_size"]:
        assert clave in params, f"falta {clave} en best_hparams.json"

    assert params["w_t1"] + params["w_t5"] == pytest.approx(1.0)
    assert (tmp_path / "hpo_trials.csv").exists()


def test_final_search_solo_usa_datos_anteriores_al_test(
    datos_sinteticos, evaluador_falso, tmp_path
):
    hpo.final_search(
        datos_sinteticos, TEST_START, n_trials=3,
        storage=f"sqlite:///{tmp_path}/f2.db", device="cpu", results_dir=tmp_path,
    )
    for llamada in evaluador_falso:
        for train_idx, val_idx in llamada["folds"]:
            assert train_idx.max() < TEST_START
            assert val_idx.max()   < TEST_START
