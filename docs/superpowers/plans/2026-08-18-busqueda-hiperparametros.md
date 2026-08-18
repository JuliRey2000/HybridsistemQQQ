# Búsqueda de Hiperparámetros con Validación Anidada — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sustituir los hiperparámetros fijos elegidos a mano por una configuración obtenida mediante búsqueda bayesiana con walk-forward anidado, sin contaminar la validación ni el test out-of-sample.

**Architecture:** Dos módulos con una frontera limpia. `src/hpo_core.py` contiene toda la lógica que **decide** (construcción de splits anidados, espacio de búsqueda, validación de configuraciones, regla de un error estándar) y depende solo de numpy — es la parte crítica para la validez de la tesis y se verifica localmente. `src/hpo.py` contiene la **maquinaria** (estudio Optuna, bucle de entrenamiento con torch, MLflow) y se ejecuta en Colab. La función de evaluación se inyecta en `run_search`, de modo que la gestión del estudio y la reanudabilidad son verificables localmente con un objetivo sintético, sin GPU.

**Tech Stack:** Python 3.10+, PyTorch 2.1.2, Optuna ≥3.4, pandas, numpy, MLflow/DagsHub, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-busqueda-hiperparametros-design.md`

## Global Constraints

- **El test OOS es intocable.** Ningún índice `>= test_start` puede aparecer en ningún split de la búsqueda, en ningún nivel. Es el requisito que invalidaría la tesis si se rompe.
- **Sin look-ahead.** En ambos niveles, todo índice de validación debe ser cronológicamente posterior a todo índice de su entrenamiento. Nunca usar splits aleatorios.
- **`config.py` no se modifica.** La configuración ganadora vive en `results/best_hparams.json`. Los valores actuales siguen siendo el default reproducible.
- **`LOOKBACK` permanece en 30.** Cambiarlo obliga a regenerar las secuencias del dataset.
- **Hiperparámetros fijos:** `num_heads=4`, `num_lstm_layers=2`, `batch_size=32`.
- **Espacio de búsqueda (6 dimensiones):** `lr` log-uniforme(1e-4, 5e-3); `weight_decay` log-uniforme(1e-6, 1e-3); `hidden_size` ∈ {64, 128, 256}; `d_model` ∈ {32, 64, 128}; `dropout` uniforme(0.1, 0.5); `w_t1` uniforme(0.3, 0.8) con `w_t5 = 1 − w_t1`.
- **Presupuesto de épocas:** 40 épocas / patience 8 durante la búsqueda; 100 épocas / patience 15 al reentrenar la ganadora.
- **Folds:** 5 externos, 3 internos, `train_min_frac=0.6` en ambos niveles.
- **Idioma:** docstrings y comentarios en español, siguiendo el estilo del repositorio.
- **El entorno local no tiene torch.** Todo test que importe `src/hpo.py`, `src/train.py` o `src/models.py` solo corre en Colab.

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `src/hpo_core.py` (crear) | Lógica pura: splits anidados, espacio de búsqueda, validación, regla 1-SE. Solo numpy + `utils.walk_forward_splits`. |
| `src/hpo.py` (crear) | Maquinaria: estudio Optuna, `evaluate_config` con torch, orquestación anidada, MLflow, estimación de costo. |
| `run_hpo.py` (crear) | Entrypoint CLI con modos `estimate`, `nested`, `final`. |
| `tests/test_hpo_core.py` (crear) | Tests de la lógica pura. Corren localmente. |
| `tests/test_hpo_search.py` (crear) | Tests del estudio Optuna con objetivo sintético inyectado. Corren localmente. |
| `requirements.txt` (modificar) | Añadir `optuna` y `pytest`. |
| `notebooks/QQQ_Hibrido_Completo.ipynb` (modificar) | Celda nueva que carga `best_hparams.json` si existe. |

---

## Task 1: Splits anidados — el núcleo de seguridad

**Files:**
- Create: `fuentes/src/hpo_core.py`
- Create: `fuentes/tests/test_hpo_core.py`
- Modify: `fuentes/requirements.txt`

**Interfaces:**
- Consumes: `utils.walk_forward_splits(n, n_splits, train_min_frac) -> list[tuple[ndarray, ndarray]]` (ya existe en `src/utils.py`)
- Produces: `InnerFold(train_idx, val_idx)`, `OuterFold(index, train_idx, val_idx, inner)`, `build_inner_folds(train_idx, n_inner=3, train_min_frac=0.6) -> tuple[InnerFold, ...]`, `build_nested_splits(test_start, n_outer=5, n_inner=3, train_min_frac=0.6) -> tuple[OuterFold, ...]`

- [ ] **Step 1: Añadir dependencias de desarrollo**

En `requirements.txt`, tras la línea `tqdm==4.66.1`, añadir:

```
# Búsqueda de hiperparámetros
optuna==3.6.1

# Testing
pytest==8.2.0
```

Instalar localmente (Python puro, no requieren GPU):

```bash
pip install optuna==3.6.1 pytest==8.2.0
```

- [ ] **Step 2: Escribir los tests que fallan**

Crear `fuentes/tests/test_hpo_core.py`:

```python
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
```

- [ ] **Step 3: Correr los tests y verificar que fallan**

Run: `cd fuentes && python -m pytest tests/test_hpo_core.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'hpo_core'`

- [ ] **Step 4: Implementar `src/hpo_core.py`**

Crear `fuentes/src/hpo_core.py`:

```python
"""
Lógica pura de la búsqueda de hiperparámetros.

Este módulo contiene todo lo que DECIDE: cómo se parten los datos, qué
configuraciones son válidas y cuál gana. Deliberadamente no importa torch ni
optuna, para que la parte crítica para la validez de la tesis pueda verificarse
en cualquier entorno, sin GPU.

La maquinaria que ejecuta la búsqueda vive en `hpo.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils import walk_forward_splits


@dataclass(frozen=True)
class InnerFold:
    """Un fold de la validación interna (la que selecciona hiperparámetros)."""
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass(frozen=True)
class OuterFold:
    """
    Un fold de la validación externa (la que estima el rendimiento).

    `inner` son los folds sobre los que corre la búsqueda de este fold externo;
    ninguno de ellos toca `val_idx`, que es lo que mantiene insesgada la
    estimación externa.
    """
    index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    inner: tuple[InnerFold, ...]


def build_inner_folds(
    train_idx: np.ndarray,
    n_inner: int = 3,
    train_min_frac: float = 0.6,
) -> tuple[InnerFold, ...]:
    """
    Construye la validación interna dentro de un conjunto de entrenamiento.

    Se reutiliza `walk_forward_splits` sobre la LONGITUD de `train_idx` y se
    mapean las posiciones resultantes a los índices reales. Así los folds
    internos heredan la garantía cronológica de la función ya probada, y el
    mapeo posicional funciona aunque `train_idx` no sea contiguo desde 0.
    """
    relativos = walk_forward_splits(
        len(train_idx), n_splits=n_inner, train_min_frac=train_min_frac
    )
    return tuple(
        InnerFold(train_idx=train_idx[tr], val_idx=train_idx[va])
        for tr, va in relativos
    )


def build_nested_splits(
    test_start: int,
    n_outer: int = 5,
    n_inner: int = 3,
    train_min_frac: float = 0.6,
) -> tuple[OuterFold, ...]:
    """
    Construye el esquema completo de walk-forward anidado.

    `test_start` acota el universo: nada por encima de ese índice entra en
    ningún nivel, que es lo que deja el test out-of-sample intacto durante toda
    la búsqueda.
    """
    externos = walk_forward_splits(
        test_start, n_splits=n_outer, train_min_frac=train_min_frac
    )
    return tuple(
        OuterFold(
            index=k,
            train_idx=tr,
            val_idx=va,
            inner=build_inner_folds(tr, n_inner=n_inner, train_min_frac=train_min_frac),
        )
        for k, (tr, va) in enumerate(externos)
    )
```

- [ ] **Step 5: Correr los tests y verificar que pasan**

Run: `cd fuentes && python -m pytest tests/test_hpo_core.py -v`
Expected: PASS — 8 tests

- [ ] **Step 6: Commit**

```bash
cd fuentes
git add src/hpo_core.py tests/test_hpo_core.py requirements.txt
git commit -m "feat: splits anidados para busqueda de hiperparametros

Nucleo de seguridad del modulo: construccion del walk-forward anidado con
garantia de que ningun indice del test OOS entra en ningun nivel de la
busqueda. Modulo puro (sin torch) para que sea verificable localmente.

Tests incluyen regresion sobre las cifras verificadas del spec: 5 folds
externos de 138 muestras de validacion y totales internos 372/414/453/495/537."
```

---

## Task 2: Espacio de búsqueda y validación de configuraciones

**Files:**
- Modify: `fuentes/src/hpo_core.py`
- Modify: `fuentes/tests/test_hpo_core.py`

**Interfaces:**
- Consumes: nada de tasks anteriores
- Produces: `FIXED_HPARAMS: dict`, `SEARCH_SPACE: dict[str, tuple]`, `complete_hparams(hp: dict) -> dict`, `suggest_hparams(trial) -> dict`, `validate_hparams(hp: dict) -> None`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir al final de `fuentes/tests/test_hpo_core.py`:

```python
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
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `cd fuentes && python -m pytest tests/test_hpo_core.py -v -k "espacio or fijos or suggest or w_t5 or combinacion or validate"`
Expected: FAIL con `ImportError: cannot import name 'FIXED_HPARAMS'`

- [ ] **Step 3: Implementar en `src/hpo_core.py`**

Añadir tras `build_nested_splits`:

```python
# ============================================================================
# ESPACIO DE BÚSQUEDA
# ============================================================================

# Fijos por decisión de diseño (ver spec, "Valores fijos y su justificación").
# num_heads=4 elimina de raíz la restricción de divisibilidad: 32, 64 y 128 son
# todos divisibles por 4, así que ninguna combinación del espacio es inválida.
FIXED_HPARAMS: dict = {
    "num_heads":       4,
    "num_lstm_layers": 2,
    "batch_size":      32,
}

# Seis dimensiones. Cada dimensión adicional es otra oportunidad de ajustar
# ruido, y con 138-537 muestras de validación el margen es estrecho.
SEARCH_SPACE: dict[str, tuple] = {
    "lr":           ("loguniform",  1e-4, 5e-3),
    "weight_decay": ("loguniform",  1e-6, 1e-3),
    "hidden_size":  ("categorical", (64, 128, 256)),
    "d_model":      ("categorical", (32, 64, 128)),
    "dropout":      ("uniform",     0.1, 0.5),
    "w_t1":         ("uniform",     0.3, 0.8),
}


def complete_hparams(hp: dict) -> dict:
    """Añade los hiperparámetros fijos y deriva w_t5 como complemento de w_t1."""
    completo = dict(FIXED_HPARAMS)
    completo.update(hp)
    completo["w_t5"] = 1.0 - completo["w_t1"]
    return completo


def suggest_hparams(trial) -> dict:
    """
    Muestrea una configuración del espacio usando un trial de Optuna.

    Acepta cualquier objeto con `suggest_float` y `suggest_categorical`, lo que
    permite probarlo sin optuna instalado.
    """
    hp: dict = {}
    for nombre, spec in SEARCH_SPACE.items():
        tipo = spec[0]
        if tipo == "loguniform":
            hp[nombre] = trial.suggest_float(nombre, spec[1], spec[2], log=True)
        elif tipo == "uniform":
            hp[nombre] = trial.suggest_float(nombre, spec[1], spec[2])
        elif tipo == "categorical":
            hp[nombre] = trial.suggest_categorical(nombre, list(spec[1]))
        else:
            raise ValueError(f"Tipo de distribución desconocido: {tipo}")
    return complete_hparams(hp)


def validate_hparams(hp: dict) -> None:
    """
    Verifica que una configuración es instanciable y coherente.

    Lanza ValueError con un mensaje explícito en vez de dejar que el fallo
    aparezca a mitad del entrenamiento.
    """
    if hp["d_model"] % hp["num_heads"] != 0:
        raise ValueError(
            f"d_model={hp['d_model']} no es divisible por num_heads={hp['num_heads']}"
        )
    if abs(hp["w_t1"] + hp["w_t5"] - 1.0) > 1e-9:
        raise ValueError(
            f"w_t1={hp['w_t1']} y w_t5={hp['w_t5']} no suman 1.0"
        )
    if not 0.0 <= hp["dropout"] < 1.0:
        raise ValueError(f"dropout={hp['dropout']} fuera del rango [0, 1)")
    if hp["lr"] <= 0:
        raise ValueError(f"lr={hp['lr']} debe ser positivo")
    if hp["hidden_size"] <= 0 or hp["d_model"] <= 0:
        raise ValueError("hidden_size y d_model deben ser positivos")
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `cd fuentes && python -m pytest tests/test_hpo_core.py -v`
Expected: PASS — 17 tests

- [ ] **Step 5: Commit**

```bash
cd fuentes
git add src/hpo_core.py tests/test_hpo_core.py
git commit -m "feat: espacio de busqueda de 6 dimensiones con validacion

Espacio reducido a proposito (lr, weight_decay, hidden_size, d_model, dropout,
w_t1) porque con 138-537 muestras de validacion cada dimension extra es otra
oportunidad de ajustar ruido. num_heads=4 fijo elimina la restriccion de
divisibilidad con d_model."
```

---

## Task 3: Regla de un error estándar

**Files:**
- Modify: `fuentes/src/hpo_core.py`
- Modify: `fuentes/tests/test_hpo_core.py`

**Interfaces:**
- Consumes: nada de tasks anteriores
- Produces: `TrialRecord(params: dict, fold_losses: tuple[float, ...])` con propiedades `mean_loss: float` y `se: float`; `select_one_se(records: list[TrialRecord]) -> TrialRecord`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir al final de `fuentes/tests/test_hpo_core.py`:

```python
from hpo_core import TrialRecord, select_one_se


def _rec(mean, hidden=128, d_model=64, dropout=0.2, wd=1e-5, spread=0.0):
    """TrialRecord con 3 folds cuya media es `mean` y dispersión controlada."""
    losses = (mean - spread, mean, mean + spread)
    params = complete_hparams({
        **VALORES_OK, "hidden_size": hidden, "d_model": d_model,
        "dropout": dropout, "weight_decay": wd,
    })
    return TrialRecord(params=params, fold_losses=losses)


def test_mean_loss_y_se_se_calculan_sobre_los_folds():
    r = _rec(1.0, spread=0.1)
    assert r.mean_loss == pytest.approx(1.0)
    assert r.se == pytest.approx(np.std([0.9, 1.0, 1.1], ddof=1) / np.sqrt(3))


def test_se_es_cero_con_un_solo_fold():
    r = TrialRecord(params=complete_hparams(VALORES_OK), fold_losses=(1.0,))
    assert r.se == 0.0


def test_elige_el_mejor_si_nadie_mas_entra_en_el_margen():
    peor  = _rec(2.0, hidden=64,  spread=0.01)
    mejor = _rec(1.0, hidden=256, spread=0.01)
    assert select_one_se([peor, mejor]) is mejor


def test_prefiere_la_config_mas_simple_dentro_de_un_error_estandar():
    """El corazón de la regla: dentro del margen gana la parsimonia."""
    mejor_crudo = _rec(1.00, hidden=256, spread=0.30)   # SE grande -> margen ancho
    mas_simple  = _rec(1.05, hidden=64,  spread=0.30)
    elegido = select_one_se([mejor_crudo, mas_simple])
    assert elegido is mas_simple
    assert elegido.params["hidden_size"] == 64


def test_desempata_por_d_model_cuando_hidden_size_empata():
    a = _rec(1.00, hidden=128, d_model=128, spread=0.30)
    b = _rec(1.02, hidden=128, d_model=32,  spread=0.30)
    assert select_one_se([a, b]) is b


def test_desempata_por_mayor_dropout_cuando_la_capacidad_empata():
    a = _rec(1.00, hidden=128, d_model=64, dropout=0.15, spread=0.30)
    b = _rec(1.02, hidden=128, d_model=64, dropout=0.45, spread=0.30)
    assert select_one_se([a, b]) is b


def test_ignora_trials_fallidos():
    fallido = TrialRecord(params=complete_hparams(VALORES_OK),
                          fold_losses=(float("inf"),) * 3)
    bueno   = _rec(1.0)
    assert select_one_se([fallido, bueno]) is bueno


def test_lanza_si_no_hay_ningun_trial_valido():
    fallido = TrialRecord(params=complete_hparams(VALORES_OK),
                          fold_losses=(float("nan"),) * 3)
    with pytest.raises(ValueError, match="válido"):
        select_one_se([fallido])
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `cd fuentes && python -m pytest tests/test_hpo_core.py -v -k "one_se or mean_loss or simple or desempata or fallidos or valido"`
Expected: FAIL con `ImportError: cannot import name 'TrialRecord'`

- [ ] **Step 3: Implementar en `src/hpo_core.py`**

Añadir al final del archivo:

```python
# ============================================================================
# SELECCIÓN: REGLA DE UN ERROR ESTÁNDAR
# ============================================================================

@dataclass(frozen=True)
class TrialRecord:
    """Resultado de un trial: su configuración y el val_loss de cada fold interno."""
    params: dict
    fold_losses: tuple[float, ...]

    @property
    def mean_loss(self) -> float:
        return float(np.mean(self.fold_losses))

    @property
    def se(self) -> float:
        """Error estándar de la media entre folds. Cero si hay menos de dos."""
        if len(self.fold_losses) < 2:
            return 0.0
        return float(np.std(self.fold_losses, ddof=1) / np.sqrt(len(self.fold_losses)))


def _parsimony_key(record: TrialRecord) -> tuple:
    """
    Orden de parsimonia del spec: menor hidden_size, menor d_model,
    mayor dropout, mayor weight_decay. Los dos últimos van negados porque
    la selección toma el mínimo.
    """
    p = record.params
    return (p["hidden_size"], p["d_model"], -p["dropout"], -p["weight_decay"])


def select_one_se(records: list[TrialRecord]) -> TrialRecord:
    """
    Elige la configuración final con la regla de un error estándar (Breiman).

    En vez de tomar el mínimo crudo de val_loss —que sobre 138-537 muestras es
    en buena parte ruido, y quedarse con el mejor de N trials equivale a tomar
    el máximo de N estimaciones ruidosas— se consideran todas las configuraciones
    dentro de un error estándar del mejor y se elige la más parsimoniosa.

    Es la práctica establecida en CART y lasso, y ataca el winner's curse
    directamente.
    """
    validos = [r for r in records if np.isfinite(r.mean_loss)]
    if not validos:
        raise ValueError("No hay ningún trial válido entre los resultados")

    mejor = min(validos, key=lambda r: r.mean_loss)
    umbral = mejor.mean_loss + mejor.se

    candidatos = [r for r in validos if r.mean_loss <= umbral]
    return min(candidatos, key=_parsimony_key)
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `cd fuentes && python -m pytest tests/test_hpo_core.py -v`
Expected: PASS — 25 tests

- [ ] **Step 5: Commit**

```bash
cd fuentes
git add src/hpo_core.py tests/test_hpo_core.py
git commit -m "feat: regla de un error estandar para seleccionar la config final

En vez del minimo crudo de val_loss, se eligen las configs dentro de 1 SE del
mejor y entre ellas la mas parsimoniosa (menor hidden_size, menor d_model,
mayor dropout, mayor weight_decay). Ataca el winner's curse, que sobre 138-537
muestras de validacion es un riesgo real."
```

---

## Task 4: Estudio Optuna con función de evaluación inyectada

**Files:**
- Create: `fuentes/src/hpo.py`
- Create: `fuentes/tests/test_hpo_search.py`

**Interfaces:**
- Consumes: `suggest_hparams`, `validate_hparams`, `TrialRecord`, `select_one_se` (Task 2 y 3)
- Produces: `run_search(evaluate_fn, n_trials, study_name, storage, seed=42) -> optuna.Study`, `records_from_study(study) -> list[TrialRecord]`

`evaluate_fn` tiene la firma `(hp: dict, trial) -> list[float]` y devuelve el `val_loss` de cada fold interno. Inyectarla es lo que permite verificar la gestión del estudio sin GPU.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `fuentes/tests/test_hpo_search.py`:

```python
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
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `cd fuentes && python -m pytest tests/test_hpo_search.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'hpo'`

- [ ] **Step 3: Implementar el esqueleto de `src/hpo.py`**

Crear `fuentes/src/hpo.py`:

```python
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

from hpo_core import TrialRecord, suggest_hparams, validate_hparams

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
        params = dict(trial.params)
        params.setdefault("w_t5", 1.0 - params.get("w_t1", 0.5))
        from hpo_core import complete_hparams
        records.append(TrialRecord(params=complete_hparams(dict(trial.params)),
                                   fold_losses=tuple(float(x) for x in losses)))
    return records
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `cd fuentes && python -m pytest tests/test_hpo_search.py -v`
Expected: PASS — 5 tests

- [ ] **Step 5: Correr la suite completa**

Run: `cd fuentes && python -m pytest tests/ -v`
Expected: PASS — 30 tests

- [ ] **Step 6: Commit**

```bash
cd fuentes
git add src/hpo.py tests/test_hpo_search.py
git commit -m "feat: gestion del estudio Optuna con evaluador inyectado

run_search recibe la funcion de evaluacion como parametro, lo que permite
verificar localmente la reanudabilidad y el manejo de fallos sin GPU.
Storage SQLite para que una desconexion de Colab no cueste el computo hecho;
un trial que revienta devuelve infinito y queda registrado sin tumbar el
estudio."
```

---

## Task 5: Evaluación con torch

**Files:**
- Modify: `fuentes/src/hpo.py`

**Interfaces:**
- Consumes: `InnerFold` (Task 1); `Trainer`, `make_dataloader` de `src/train.py`; `HybridPredictiveModel` de `src/models.py`; `scale_price_sequences` de `src/utils.py`
- Produces: `evaluate_config(hp, data, inner_folds, device, epochs=40, patience=8, trial=None, seed=42) -> list[float]`, `estimate_cost(data, outer_folds, device) -> dict`

Esta task no es verificable localmente (requiere torch). Su verificación es el smoke run de la Task 7 en Colab.

- [ ] **Step 1: Implementar `evaluate_config`**

Añadir a `fuentes/src/hpo.py`, tras `records_from_study`:

```python
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

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            ckpt = tmp.name

        history = trainer.fit(
            train_loader, val_loader,
            epochs=epochs, patience=patience, save_path=ckpt,
        )
        losses.append(float(min(history["val_loss"])))

        del model, trainer
        if device == "cuda":
            torch.cuda.empty_cache()

        if trial is not None:
            trial.report(float(np.mean(losses)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return losses
```

- [ ] **Step 2: Implementar `estimate_cost`**

Añadir a continuación:

```python
def estimate_cost(data: dict, outer_folds, device: str, n_trials: int = 40) -> dict:
    """
    Cronometra UN trial real con la configuración por defecto y extrapola.

    El spec estima ~15h sin medir; esta función da el número real antes de
    comprometer horas de GPU.
    """
    import time

    from hpo_core import complete_hparams

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
```

- [ ] **Step 3: Verificar que el módulo sigue compilando**

Run: `cd fuentes && python -m py_compile src/hpo.py && echo OK`
Expected: `OK` (compila; la ejecución real requiere torch)

- [ ] **Step 4: Verificar que los tests locales siguen pasando**

Run: `cd fuentes && python -m pytest tests/ -v`
Expected: PASS — 30 tests (los imports de torch están dentro de las funciones, así que no rompen los tests locales)

- [ ] **Step 5: Commit**

```bash
cd fuentes
git add src/hpo.py
git commit -m "feat: evaluacion de configuraciones con torch + estimacion de costo

evaluate_config entrena en los 3 folds internos reajustando el scaler con el
train de cada uno, y reporta al pruner tras cada fold para abandonar configs
malas sin completar los tres. Los imports de torch van dentro de la funcion
para no romper los tests locales, que corren sin GPU.

estimate_cost cronometra un trial real y extrapola, porque las ~15h del spec
son una estimacion sin medir."
```

---

## Task 6: Orquestación anidada y CLI

**Files:**
- Modify: `fuentes/src/hpo.py`
- Create: `fuentes/run_hpo.py`

**Interfaces:**
- Consumes: `build_nested_splits` (Task 1), `run_search`/`records_from_study` (Task 4), `evaluate_config`/`estimate_cost` (Task 5), `select_one_se` (Task 3)
- Produces: `nested_walk_forward(data, test_start, n_trials, storage, device, results_dir) -> dict`, `final_search(data, test_start, n_trials, storage, device, results_dir) -> dict`

- [ ] **Step 1: Implementar la orquestación**

Añadir al final de `fuentes/src/hpo.py`:

```python
# ============================================================================
# ORQUESTACIÓN
# ============================================================================

def _buscar_sobre(data, inner_folds, n_trials, study_name, storage, device):
    """Corre una búsqueda sobre unos folds internos y devuelve la config elegida."""
    from hpo_core import select_one_se

    def evaluador(hp, trial):
        return evaluate_config(hp, data, inner_folds, device, trial=trial)

    study = run_search(evaluador, n_trials=n_trials,
                       study_name=study_name, storage=storage)
    records = records_from_study(study)
    elegido = select_one_se(records)
    return elegido, records


def nested_walk_forward(
    data: dict,
    test_start: int,
    n_trials: int,
    storage: str,
    device: str,
    results_dir,
) -> dict:
    """
    Protocolo anidado completo: la estimación insesgada del procedimiento.

    Por cada fold externo corre una búsqueda dentro de su propio train, reentrena
    la ganadora sobre el train externo completo y la evalúa en la validación
    externa, que nunca participó en la selección.
    """
    import json

    import pandas as pd

    from hpo_core import build_nested_splits
    from utils import predictive_metrics, scale_price_sequences

    outer_folds = build_nested_splits(test_start)
    filas = []

    for outer in outer_folds:
        logger.info(f"Fold externo {outer.index}: búsqueda sobre {len(outer.inner)} folds internos")
        elegido, _ = _buscar_sobre(
            data, outer.inner, n_trials,
            study_name=f"nested_outer{outer.index}", storage=storage, device=device,
        )

        # Reentrenar la ganadora con el presupuesto completo y evaluar fuera
        from hpo_core import InnerFold
        fold_externo = (InnerFold(train_idx=outer.train_idx, val_idx=outer.val_idx),)
        val_loss = evaluate_config(
            elegido.params, data, fold_externo, device, epochs=100, patience=15,
        )[0]

        fila = {"fold": outer.index, "outer_val_loss": val_loss, **elegido.params}
        filas.append(fila)
        logger.info(f"  config elegida -> val_loss externo {val_loss:.6f}")

    df = pd.DataFrame(filas)
    df.to_csv(results_dir / "hpo_nested.csv", index=False)

    resumen = {
        "outer_val_loss_media": float(df["outer_val_loss"].mean()),
        "outer_val_loss_std":   float(df["outer_val_loss"].std()),
        "folds": filas,
    }
    (results_dir / "hpo_nested_resumen.json").write_text(
        json.dumps(resumen, indent=2), encoding="utf-8"
    )
    return resumen


def final_search(
    data: dict,
    test_start: int,
    n_trials: int,
    storage: str,
    device: str,
    results_dir,
) -> dict:
    """
    Búsqueda final sobre todo el train+val. Produce la config reportable.

    Usa los folds internos del último fold externo, que son los que cubren el
    mayor rango de datos disponible sin tocar el test.
    """
    import json

    import pandas as pd

    from hpo_core import build_inner_folds

    inner = build_inner_folds(np.arange(0, test_start))
    elegido, records = _buscar_sobre(
        data, inner, n_trials,
        study_name="final", storage=storage, device=device,
    )

    pd.DataFrame([
        {"mean_val_loss": r.mean_loss, "se": r.se, **r.params} for r in records
    ]).to_csv(results_dir / "hpo_trials.csv", index=False)

    (results_dir / "best_hparams.json").write_text(
        json.dumps(elegido.params, indent=2), encoding="utf-8"
    )
    logger.info(f"Config final guardada en {results_dir / 'best_hparams.json'}")
    return elegido.params
```

- [ ] **Step 2: Crear el entrypoint `run_hpo.py`**

Crear `fuentes/run_hpo.py`:

```python
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


def cargar_datos() -> dict:
    """Carga las secuencias ya preprocesadas por run_pipeline.py."""
    req = ["price_seqs.npy", "sentiments.npy", "y_t1.npy", "y_t5.npy"]
    faltan = [f for f in req if not (DATA_PROCESSED_PATH / f).exists()]
    if faltan:
        raise FileNotFoundError(
            f"Faltan {faltan} en {DATA_PROCESSED_PATH}.\n"
            "Ejecuta primero: python run_pipeline.py"
        )
    return {
        "price_seqs": np.load(DATA_PROCESSED_PATH / "price_seqs.npy"),
        "sentiments": np.load(DATA_PROCESSED_PATH / "sentiments.npy"),
        "y_t1":       np.load(DATA_PROCESSED_PATH / "y_t1.npy"),
        "y_t5":       np.load(DATA_PROCESSED_PATH / "y_t5.npy"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["estimate", "nested", "final"], required=True)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--inner-folds", type=int, default=3)
    args = ap.parse_args()

    import hpo

    data = cargar_datos()
    n = len(data["price_seqs"])
    test_start, _ = final_test_split(n, TEST_FRAC)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{RESULTS_PATH / 'hpo.db'}"

    logger.info(f"Muestras: {n}  |  test_start: {test_start}  |  dispositivo: {DEVICE}")

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
        return 0

    if args.mode == "nested":
        resumen = hpo.nested_walk_forward(
            data, test_start, args.trials, storage, DEVICE, RESULTS_PATH
        )
        print(f"\nval_loss externo medio: {resumen['outer_val_loss_media']:.6f} "
              f"± {resumen['outer_val_loss_std']:.6f}")
        return 0

    params = hpo.final_search(
        data, test_start, args.trials, storage, DEVICE, RESULTS_PATH
    )
    print("\nConfiguración final:")
    for k, v in sorted(params.items()):
        print(f"  {k:16s}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verificar que compila y que el CLI responde**

Run: `cd fuentes && python -m py_compile run_hpo.py src/hpo.py && python run_hpo.py --help`
Expected: se imprime la ayuda con los tres modos, sin error

- [ ] **Step 4: Verificar que los tests locales siguen pasando**

Run: `cd fuentes && python -m pytest tests/ -v`
Expected: PASS — 30 tests

- [ ] **Step 5: Commit**

```bash
cd fuentes
git add src/hpo.py run_hpo.py
git commit -m "feat: orquestacion anidada y CLI run_hpo.py

nested_walk_forward corre una busqueda por fold externo y evalua la ganadora en
la validacion externa, que nunca participo en la seleccion. final_search produce
results/best_hparams.json. CLI con modos estimate/nested/final; estimate se corre
siempre primero para medir el costo real antes de comprometer horas de GPU."
```

---

## Task 7: Integración con el notebook y verificación en Colab

**Files:**
- Modify: `fuentes/notebooks/QQQ_Hibrido_Completo.ipynb`
- Modify: `fuentes/PROGRESS.md`

**Interfaces:**
- Consumes: `results/best_hparams.json` producido por Task 6
- Produces: la celda del notebook que alimenta `MODEL_HP` y `TRAIN_HP`

- [ ] **Step 1: Añadir la celda de carga de configuración**

Insertar una celda de código nueva **inmediatamente después de la celda 12** (la que define `MODEL_HP`/`TRAIN_HP`), con `id` = `cell-load-hparams`:

```python
# ─── Cargar configuración optimizada si existe ───────────────────────────────
# Si se corrió run_hpo.py, best_hparams.json reemplaza los valores por defecto.
# Sin ese archivo el notebook sigue funcionando con la config de config.py, así
# que esta celda nunca lo rompe.
import json
from pathlib import Path

HPARAMS_JSON = Path('results/best_hparams.json')

if HPARAMS_JSON.exists():
    _best = json.loads(HPARAMS_JSON.read_text())
    MODEL_HP.update({
        'hidden_size':     _best['hidden_size'],
        'd_model':         _best['d_model'],
        'num_heads':       _best['num_heads'],
        'num_lstm_layers': _best['num_lstm_layers'],
        'dropout':         _best['dropout'],
    })
    TRAIN_HP.update({
        'lr':           _best['lr'],
        'weight_decay': _best['weight_decay'],
        'w_t1':         _best['w_t1'],
        'w_t5':         _best['w_t5'],
        'batch_size':   _best['batch_size'],
    })
    HPARAMS_SOURCE = 'busqueda_optuna'
    print('✓ Config optimizada cargada desde results/best_hparams.json')
    for k, v in sorted(_best.items()):
        print(f'    {k:16s}: {v}')
else:
    HPARAMS_SOURCE = 'defaults_config_py'
    print('ℹ Sin best_hparams.json — usando los valores por defecto de config.py')
    print('  Para optimizarlos: python run_hpo.py --mode estimate')

print(f'\nMODEL_HP: {MODEL_HP}')
print(f'TRAIN_HP: {TRAIN_HP}')
```

- [ ] **Step 2: Registrar la procedencia de la config en MLflow**

En la celda 13 (`cell-wf-train`), dentro del `mlflow.log_params({...})`, añadir la clave `'hparams_source': HPARAMS_SOURCE` junto a `'feature_scaling'`. Así cada run de DagsHub queda marcado como "con búsqueda" o "con defaults", que es lo que permite comparar antes/después en el documento.

- [ ] **Step 3: Validar el notebook**

Run:
```bash
cd fuentes && python -c "
import json, pathlib
nb = json.loads(pathlib.Path('notebooks/QQQ_Hibrido_Completo.ipynb').read_text(encoding='utf-8'))
ids = [c.get('metadata', {}).get('id', '') for c in nb['cells']]
assert 'cell-load-hparams' in ids, 'falta la celda nueva'
import ast
for c in nb['cells']:
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        if not src.strip().startswith(('%', '!')):
            ast.parse(src)
print(f'OK: {len(nb[\"cells\"])} celdas, sintaxis valida, celda nueva presente')
"
```
Expected: `OK: 30 celdas, sintaxis valida, celda nueva presente`

- [ ] **Step 4: Correr la suite completa una última vez**

Run: `cd fuentes && python -m pytest tests/ -v`
Expected: PASS — 30 tests

- [ ] **Step 5: Actualizar PROGRESS.md**

Añadir una sección al principio, bajo el encabezado de estado, describiendo: el módulo implementado, que la ejecución espera al corpus FinBERT, y el orden de comandos (`estimate` → `nested` → `final`).

- [ ] **Step 6: Commit y push**

```bash
cd fuentes
git add notebooks/QQQ_Hibrido_Completo.ipynb PROGRESS.md
git commit -m "feat: el notebook carga la config optimizada si existe

Celda nueva tras la definicion de MODEL_HP/TRAIN_HP que lee
results/best_hparams.json y sobrescribe los valores por defecto. Sin ese archivo
el notebook funciona igual que hoy, asi que el cambio no rompe nada.

MLflow registra hparams_source para poder comparar en el documento los runs con
busqueda contra los de defaults."
git push origin main
```

- [ ] **Step 7: Smoke run en Colab (requiere GPU — lo ejecuta el usuario)**

Este es el único paso que no se puede verificar localmente. En Colab, tras `git pull`:

```bash
!cd /content/HybridsistemQQQ && pip install optuna==3.6.1 -q
!cd /content/HybridsistemQQQ && python run_hpo.py --mode estimate --trials 2
```

Expected: imprime la tabla de estimación de costo sin errores, con `segundos_por_trial` > 0.

Si falla, el error más probable es un desajuste en los nombres de los `.npy` que produce `run_pipeline.py`; verificar contra `cargar_datos()` en `run_hpo.py`.

---

## Self-Review

**Cobertura del spec:**

| Requisito del spec | Task |
|---|---|
| Protocolo anidado, test intocable | 1 |
| Walk-forward interno de 3 folds | 1 |
| Escalado sin fuga por nivel | 5 |
| Espacio de 6 dimensiones + fijos | 2 |
| Regla de un error estándar | 3 |
| Storage SQLite reanudable | 4 |
| Pruning | 4 (configuración), 5 (report por fold) |
| Trials fallidos → inf | 4 |
| `best_hparams.json`, `hpo_trials.csv`, `hpo_nested.csv` | 6 |
| `estimate_cost` antes de comprometer horas | 5, 6 |
| Los 7 tests del spec | 1 (no contaminación, no look-ahead, aislamiento), 2 (validez del espacio), 3 (regla 1-SE), 4 (reanudabilidad), 7 (smoke run) |
| Integración con el notebook | 7 |
| MLflow | 7 (`hparams_source`); el logging por trial queda en el estudio SQLite |

**Nota de alcance:** el spec menciona runs anidados de MLflow por trial. Se sustituye por el storage SQLite de Optuna como registro primario (`hpo_trials.csv` es el artefacto para el documento) y una marca `hparams_source` en el run de entrenamiento. Registrar 240 runs en DagsHub aportaría ruido sin valor para la tesis. Si se quiere el detalle en MLflow, es una adición posterior de una función.

**Consistencia de tipos:** `evaluate_fn(hp, trial) -> list[float]` en Tasks 4, 5 y 6. `InnerFold`/`OuterFold` con los mismos campos en 1, 5 y 6. `TrialRecord(params, fold_losses)` en 3, 4 y 6. `select_one_se(records) -> TrialRecord` en 3 y 6.
