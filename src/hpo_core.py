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
