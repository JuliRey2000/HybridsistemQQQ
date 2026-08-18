"""
Tests de la lógica pura de búsqueda de hiperparámetros.

No importan torch: corren en cualquier entorno con numpy.
El primer test es el más importante del proyecto — si el test OOS se filtra
en la búsqueda, todos los resultados de la tesis quedan invalidados.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from hpo_core import build_inner_folds, build_nested_splits

TEST_START = 2068   # el corte real del proyecto: int(2433 * 0.85)


def test_ningun_indice_de_test_entra_en_la_busqueda():
    """EL test crítico: nada >= test_start puede aparecer en ningún nivel."""
    for outer in build_nested_splits(TEST_START):
        assert outer.train_idx.max() < TEST_START
        assert outer.val_idx.max()   < TEST_START
        for inner in outer.inner:
            assert inner.train_idx.max() < TEST_START
            assert inner.val_idx.max()   < TEST_START


def test_sin_lookahead_en_el_nivel_externo():
    for outer in build_nested_splits(TEST_START):
        assert outer.train_idx.max() < outer.val_idx.min()


def test_sin_lookahead_en_el_nivel_interno():
    for outer in build_nested_splits(TEST_START):
        for inner in outer.inner:
            assert inner.train_idx.max() < inner.val_idx.min()


def test_la_busqueda_interna_nunca_toca_la_validacion_externa():
    """Si el inner viera el outer_val, la estimación externa dejaría de ser insesgada."""
    for outer in build_nested_splits(TEST_START):
        prohibidos = set(outer.val_idx.tolist())
        for inner in outer.inner:
            assert not (set(inner.train_idx.tolist()) & prohibidos)
            assert not (set(inner.val_idx.tolist())   & prohibidos)


def test_los_folds_internos_viven_dentro_del_train_externo():
    for outer in build_nested_splits(TEST_START):
        permitidos = set(outer.train_idx.tolist())
        for inner in outer.inner:
            assert set(inner.train_idx.tolist()) <= permitidos
            assert set(inner.val_idx.tolist())   <= permitidos


def test_tamanos_coinciden_con_la_tabla_del_spec():
    """Regresión sobre las cifras verificadas que documenta el spec."""
    folds = build_nested_splits(TEST_START)
    assert len(folds) == 5
    assert [len(f.val_idx) for f in folds] == [138] * 5
    assert [len(f.train_idx) for f in folds] == [1240, 1378, 1516, 1654, 1792]
    totales_internos = [sum(len(i.val_idx) for i in f.inner) for f in folds]
    assert totales_internos == [372, 414, 453, 495, 537]


def test_cada_fold_externo_tiene_tres_folds_internos():
    for outer in build_nested_splits(TEST_START):
        assert len(outer.inner) == 3


def test_build_inner_folds_respeta_indices_no_contiguos():
    """Los índices se mapean por posición, no se asumen contiguos desde 0."""
    train_idx = np.arange(100, 300)
    folds = build_inner_folds(train_idx, n_inner=3)
    for f in folds:
        assert f.train_idx.min() >= 100
        assert f.val_idx.max()   < 300
        assert f.train_idx.max() < f.val_idx.min()


# ============================================================================
# ESPACIO DE BÚSQUEDA
# ============================================================================

from hpo_core import (
    FIXED_HPARAMS, SEARCH_SPACE,
    complete_hparams, suggest_hparams, validate_hparams,
)


class FakeTrial:
    """Sustituto de optuna.Trial para probar el muestreo sin instalar optuna."""

    def __init__(self, valores: dict):
        self.valores = valores
        self.pedidos: list[str] = []

    def suggest_float(self, name, low, high, log=False):
        self.pedidos.append(name)
        assert low <= self.valores[name] <= high, f"{name} fuera de rango"
        return self.valores[name]

    def suggest_categorical(self, name, choices):
        self.pedidos.append(name)
        assert self.valores[name] in choices, f"{name} fuera del conjunto"
        return self.valores[name]


VALORES_OK = {
    "lr": 1e-3, "weight_decay": 1e-5, "hidden_size": 128,
    "d_model": 64, "dropout": 0.2, "w_t1": 0.6,
}


def test_el_espacio_tiene_exactamente_seis_dimensiones():
    assert len(SEARCH_SPACE) == 6
    assert set(SEARCH_SPACE) == {
        "lr", "weight_decay", "hidden_size", "d_model", "dropout", "w_t1"
    }


def test_los_fijos_son_los_del_spec():
    assert FIXED_HPARAMS == {"num_heads": 4, "num_lstm_layers": 2, "batch_size": 32}


def test_suggest_consulta_las_seis_dimensiones():
    trial = FakeTrial(VALORES_OK)
    hp = suggest_hparams(trial)
    assert set(trial.pedidos) == set(SEARCH_SPACE)
    assert hp["hidden_size"] == 128


def test_w_t5_es_el_complemento_de_w_t1():
    hp = complete_hparams({**VALORES_OK})
    assert hp["w_t1"] + hp["w_t5"] == pytest.approx(1.0)


def test_los_fijos_se_incorporan_a_la_config():
    hp = complete_hparams({**VALORES_OK})
    for k, v in FIXED_HPARAMS.items():
        assert hp[k] == v


def test_toda_combinacion_categorica_del_espacio_es_valida():
    """num_heads=4 fijo hace que 32, 64 y 128 sean siempre divisibles."""
    for hidden in SEARCH_SPACE["hidden_size"][1]:
        for d_model in SEARCH_SPACE["d_model"][1]:
            hp = complete_hparams({**VALORES_OK, "hidden_size": hidden, "d_model": d_model})
            validate_hparams(hp)   # no debe lanzar


def test_validate_rechaza_d_model_no_divisible():
    hp = complete_hparams({**VALORES_OK})
    hp["d_model"] = 33
    with pytest.raises(ValueError, match="divisible"):
        validate_hparams(hp)


def test_validate_rechaza_pesos_que_no_suman_uno():
    hp = complete_hparams({**VALORES_OK})
    hp["w_t5"] = 0.9
    with pytest.raises(ValueError, match="suman"):
        validate_hparams(hp)


def test_validate_rechaza_dropout_fuera_de_rango():
    hp = complete_hparams({**VALORES_OK})
    hp["dropout"] = 1.5
    with pytest.raises(ValueError, match="dropout"):
        validate_hparams(hp)
