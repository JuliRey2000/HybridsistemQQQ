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
