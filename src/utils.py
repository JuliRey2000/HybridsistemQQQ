"""
Utilidades de evaluación, métricas y visualización.

Implementa las métricas exigidas por la tesis:
  PREDICTIVO  : RMSE, Directional Accuracy, Sharpe Ratio
  GENERATIVO  : Wasserstein Distance (EMD), Stylized Facts
                (volatility clustering, leverage effect)
  BACKTESTING : Estrategia long/short, Max Drawdown, Sortino Ratio
"""

from __future__ import annotations

import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ============================================================================
# SPLIT WALK-FORWARD (sustituto del split estático 70/15/15)
# ============================================================================

def walk_forward_splits(
    n: int,
    n_splits: int = 5,
    train_min_frac: float = 0.6,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """
    Genera splits walk-forward para validación cronológica.

    En cada split, el conjunto de entrenamiento crece acumulativamente
    y el conjunto de validación siempre es posterior al de entrenamiento.
    Esto simula el uso real del modelo en producción.

    Ejemplo con n=1000 y n_splits=5:
      Split 0: train=[0..599],  val=[600..699]
      Split 1: train=[0..699],  val=[700..799]
      Split 2: train=[0..799],  val=[800..899]
      ...

    Args:
        n             : número total de muestras
        n_splits      : número de folds (default: 5)
        train_min_frac: fracción mínima para entrenamiento inicial (default: 0.6)

    Returns:
        Lista de (train_indices, val_indices) arrays
    """
    min_train = int(n * train_min_frac)
    val_size  = (n - min_train) // (n_splits + 1)

    splits = []
    for k in range(n_splits):
        train_end = min_train + k * val_size
        val_end   = train_end + val_size
        if val_end > n:
            break
        train_idx = np.arange(0, train_end)
        val_idx   = np.arange(train_end, val_end)
        splits.append((train_idx, val_idx))

    logger.info(f"Walk-forward: {len(splits)} splits, val_size≈{val_size}")
    return splits


def final_test_split(n: int, test_frac: float = 0.15) -> Tuple[int, int]:
    """
    Reserva el `test_frac` final como test out-of-sample.

    Returns:
        (train_val_end, test_start) — índices de corte
    """
    test_start = int(n * (1 - test_frac))
    return test_start, n


# ============================================================================
# NORMALIZACIÓN DE SECUENCIAS (sin data leakage)
# ============================================================================

def scale_price_sequences(
    price_seqs: np.ndarray,
    fit_idx: np.ndarray,
) -> Tuple[np.ndarray, "object"]:
    """
    Normaliza las secuencias de features con StandardScaler ajustado SOLO
    sobre las ventanas de entrenamiento (sin data leakage temporal).

    Crítico para este dataset: SMA_20/SMA_50 y ATR están en USD (~100-530 y
    crecientes 2015→2024), RSI en [0,100], BB_Pct en [0,1]. Sin normalizar,
    el LSTM recibe escalas mezcladas y el test set queda fuera de la
    distribución de entrenamiento (los precios de 2024 superan todo lo visto).

    Args:
        price_seqs: (n, lookback, features) — secuencias completas
        fit_idx   : índices de las muestras de entrenamiento (el scaler solo
                    ve días <= último día de entrenamiento)

    Returns:
        (scaled, scaler):
          scaled — array completo (n, lookback, features) transformado
          scaler — StandardScaler ajustado (para transformar datos futuros)
    """
    from sklearn.preprocessing import StandardScaler

    n, lookback, n_features = price_seqs.shape
    scaler = StandardScaler().fit(price_seqs[fit_idx].reshape(-1, n_features))
    scaled = (
        scaler.transform(price_seqs.reshape(-1, n_features))
        .reshape(n, lookback, n_features)
        .astype(np.float32)
    )
    return scaled, scaler


def transform_price_sequences(price_seqs: np.ndarray, scaler) -> np.ndarray:
    """Aplica un scaler ya ajustado a secuencias (n, lookback, features)."""
    n, lookback, n_features = price_seqs.shape
    return (
        scaler.transform(price_seqs.reshape(-1, n_features))
        .reshape(n, lookback, n_features)
        .astype(np.float32)
    )


# ============================================================================
# MÉTRICAS PREDICTIVAS
# ============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precisión Direccional: fracción de días donde el modelo predice
    correctamente si el mercado sube o baja.

    DA = count(sign(y_pred) == sign(y_true)) / n

    Benchmark aleatorio: 0.50 (50 %)
    """
    correct = np.sign(y_pred) == np.sign(y_true)
    return float(correct.mean())


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe Ratio anualizado.

    SR = (μ_retorno - r_f) / σ_retorno * √(períodos_año)

    Args:
        returns          : retornos de la estrategia (en %, no decimal)
        risk_free_rate   : tasa anual libre de riesgo (default: 2%)
        periods_per_year : días de trading (default: 252)
    """
    r = returns / 100.0
    excess = r - risk_free_rate / periods_per_year
    if np.std(excess) < 1e-9:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Sortino Ratio: penaliza solo la volatilidad a la baja (más conservador).
    """
    r = returns / 100.0
    excess = r - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-9
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """
    Máxima caída acumulada desde un pico hasta un valle.

    Returns:
        float: porcentaje de caída máxima (negativo)
    """
    r = returns / 100.0
    cumulative = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative / peak - 1) * 100
    return float(np.min(drawdown))


def predictive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compila todas las métricas predictivas en un diccionario."""
    return {
        "RMSE":                 rmse(y_true, y_pred),
        "MAE":                  mae(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
    }


# ============================================================================
# SIGNIFICANCIA ESTADÍSTICA (Pesaran-Timmermann, Diebold-Mariano)
# ============================================================================

def pesaran_timmermann(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Test de Pesaran-Timmermann (1992) de capacidad predictiva direccional.

    H0: los signos de predicción y realización son independientes (el modelo
    no tiene información direccional). Bajo H0 el estadístico es ~N(0,1).
    p-value unilateral: P(DA observada > DA esperada bajo independencia).

    Nota: el test asume observaciones independientes. Para el horizonte t+5
    (retornos acumulados solapados con paso diario) los targets están
    autocorrelacionados y el p-value es aproximado (tiende a ser optimista).

    Args:
        y_true: retornos realizados (%)
        y_pred: retornos predichos (%)

    Returns:
        dict: da (observada), da_indep (esperada bajo H0), stat, p_value
    """
    y = np.asarray(y_true, dtype=float).flatten()
    x = np.asarray(y_pred, dtype=float).flatten()
    n = len(y)

    p_hat = float(np.mean(np.sign(x) == np.sign(y)))
    py = float(np.mean(y > 0))
    px = float(np.mean(x > 0))
    p_star = py * px + (1 - py) * (1 - px)

    var_p_hat  = p_star * (1 - p_star) / n
    var_p_star = (
        (2 * py - 1) ** 2 * px * (1 - px) / n
        + (2 * px - 1) ** 2 * py * (1 - py) / n
        + 4 * py * px * (1 - py) * (1 - px) / n ** 2
    )

    denom = var_p_hat - var_p_star
    if denom <= 0:
        # Degenera cuando las predicciones son casi todas del mismo signo
        # (var_p_hat ≈ var_p_star): el test no es informativo
        return {"da": p_hat, "da_indep": p_star, "stat": float("nan"),
                "p_value": float("nan"), "n": n}

    stat = (p_hat - p_star) / np.sqrt(denom)
    p_value = float(stats.norm.sf(stat))
    return {"da": p_hat, "da_indep": p_star, "stat": float(stat),
            "p_value": p_value, "n": n}


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    h: int = 1,
    loss: str = "mse",
) -> dict:
    """
    Test de Diebold-Mariano (1995) con corrección de muestra pequeña de
    Harvey-Leybourne-Newbold (1997).

    Compara la pérdida de dos pronósticos: d_t = L(e_a,t) − L(e_b,t).
    stat < 0 → el modelo A tiene menor pérdida que el B.
    p-value unilateral (H1: A es mejor que B), contra t de Student (n−1 gl).

    La varianza de largo plazo de d_t usa la ventana rectangular con h−1
    rezagos del artículo original — necesario para h>1 (pronósticos de
    horizonte multi-día con errores solapados autocorrelacionados).

    Args:
        y_true: valores realizados (%)
        pred_a: pronóstico del modelo A (el que se postula mejor)
        pred_b: pronóstico del modelo B (benchmark)
        h     : horizonte de pronóstico (1 para t+1, 5 para t+5)
        loss  : 'mse' (error cuadrático) o 'mae' (error absoluto)

    Returns:
        dict: stat (HLN-corregido), p_value (unilateral), mean_d
    """
    y = np.asarray(y_true, dtype=float).flatten()
    e_a = y - np.asarray(pred_a, dtype=float).flatten()
    e_b = y - np.asarray(pred_b, dtype=float).flatten()

    if loss == "mse":
        d = e_a ** 2 - e_b ** 2
    elif loss == "mae":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError(f"loss desconocida: {loss!r} (usar 'mse' o 'mae')")

    n = len(d)
    d_bar = float(d.mean())
    d_c = d - d_bar

    lr_var = float(np.mean(d_c ** 2))          # gamma_0
    for lag in range(1, h):
        gamma = float(np.mean(d_c[lag:] * d_c[:-lag]))
        lr_var += 2 * gamma
    if lr_var <= 0:
        # La ventana rectangular puede dar varianza negativa; fallback a gamma_0
        lr_var = float(np.mean(d_c ** 2))
    if lr_var <= 0:
        # Pronósticos idénticos (d_t = 0 para todo t): el test no aplica
        return {"stat": float("nan"), "p_value": float("nan"),
                "mean_d": d_bar, "n": n}

    dm_stat = d_bar / np.sqrt(lr_var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(hln * dm_stat)

    p_value = float(stats.t.cdf(stat, df=n - 1))   # unilateral: A mejor (stat<0)
    return {"stat": stat, "p_value": p_value, "mean_d": d_bar, "n": n}


# ============================================================================
# MÉTRICAS GENERATIVAS
# ============================================================================

def wasserstein_distance_1d(real: np.ndarray, fake: np.ndarray) -> float:
    """
    Distancia de Wasserstein (EMD, Earth Mover's Distance) en 1D.

    Mide cuánto 'esfuerzo' se necesita para transformar la distribución
    de retornos generados en la distribución real.

    W = integral |CDF_real(x) - CDF_fake(x)| dx

    Valor = 0  →  distribuciones idénticas
    Valor → ∞  →  distribuciones muy distintas

    Args:
        real: retornos reales (1D array)
        fake: retornos generados (1D array)
    """
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(real.flatten(), fake.flatten()))


def stylized_facts(returns: np.ndarray, label: str = "serie") -> dict:
    """
    Verifica los 'hechos estilizados' de series financieras.

    Hechos comprobados:
      1. Agrupación de volatilidad (volatility clustering):
         correlación positiva entre |r_t| y |r_{t-1}|
      2. Efecto apalancamiento (leverage effect):
         correlación negativa entre r_t y σ_{t+k} (retorno actual vs
         volatilidad futura) — capturado por correlación r vs |r| rezagado
      3. Colas gruesas (heavy tails): kurtosis excesiva > 3
      4. Asimetría negativa (negative skewness): bajadas más bruscas

    Args:
        returns: retornos diarios (array 1D o 2D — promedia columnas si 2D)
        label  : etiqueta para logging

    Returns:
        dict con métricas de hechos estilizados
    """
    r = returns.flatten() if returns.ndim > 1 else returns

    # Agrupación de volatilidad: autocorrelación de |retorno|
    abs_r = np.abs(r)
    if len(abs_r) > 2:
        vol_clustering, _ = spearmanr(abs_r[:-1], abs_r[1:])
    else:
        vol_clustering = 0.0

    # Efecto apalancamiento: correlación retorno vs volatilidad futura
    if len(r) > 5:
        leverage, _ = spearmanr(r[:-5], abs_r[5:])
    else:
        leverage = 0.0

    kurtosis = float(stats.kurtosis(r, fisher=True))   # exceso (normal=0)
    skewness = float(stats.skew(r))

    result = {
        "volatility_clustering": float(vol_clustering),
        "leverage_effect":       float(leverage),
        "excess_kurtosis":       kurtosis,
        "skewness":              skewness,
    }

    logger.info(
        f"[Hechos estilizados — {label}] "
        f"Vol.Clustering: {vol_clustering:.4f} | "
        f"Leverage: {leverage:.4f} | "
        f"Kurtosis: {kurtosis:.4f} | "
        f"Skewness: {skewness:.4f}"
    )
    return result


def generative_metrics(
    real_returns: np.ndarray,
    fake_returns: np.ndarray,
) -> dict:
    """
    Compila métricas de evaluación del módulo generativo.

    Args:
        real_returns: trayectorias reales  (n_samples, seq_len)
        fake_returns: trayectorias generadas (n_samples, seq_len)
    """
    w_dist = wasserstein_distance_1d(real_returns, fake_returns)
    sf_real = stylized_facts(real_returns, label="Real")
    sf_fake = stylized_facts(fake_returns, label="Generado")

    return {
        "wasserstein_distance":          w_dist,
        "real_vol_clustering":           sf_real["volatility_clustering"],
        "fake_vol_clustering":           sf_fake["volatility_clustering"],
        "real_leverage":                 sf_real["leverage_effect"],
        "fake_leverage":                 sf_fake["leverage_effect"],
        "real_excess_kurtosis":          sf_real["excess_kurtosis"],
        "fake_excess_kurtosis":          sf_fake["excess_kurtosis"],
    }


# ============================================================================
# BACKTESTING
# ============================================================================

def long_short_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.0,
) -> dict:
    """
    Simula estrategia long/short basada en predicciones.

    Señal:
      pred > +threshold  → largo (comprar)
      pred < -threshold  → corto (vender)
      en rango           → sin posición

    Args:
        y_true    : retornos reales (%)
        y_pred    : predicciones (%)
        threshold : umbral de señal (default: 0.0 %)

    Returns:
        dict con retornos de estrategia y métricas de cartera
    """
    signals = np.zeros(len(y_pred))
    signals[y_pred >  threshold] =  1.0
    signals[y_pred < -threshold] = -1.0

    strategy_returns = signals * y_true   # retorno diario de la estrategia
    bh_returns       = y_true             # buy & hold como benchmark

    metrics = {
        "strategy_sharpe":       sharpe_ratio(strategy_returns),
        "strategy_sortino":      sortino_ratio(strategy_returns),
        "strategy_max_drawdown": max_drawdown(strategy_returns),
        "strategy_total_return": float(strategy_returns.sum()),
        "strategy_directional":  directional_accuracy(y_true, y_pred),
        "bh_sharpe":             sharpe_ratio(bh_returns),
        "bh_total_return":       float(bh_returns.sum()),
        "num_trades":            int(np.sum(signals != 0)),
        "strategy_returns":      strategy_returns,
        "bh_returns":            bh_returns,
    }

    return metrics


# ============================================================================
# VISUALIZACIONES
# ============================================================================

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: str = "t+1",
    save_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(y_true, label="Real", color="navy", linewidth=1.2, alpha=0.8)
    axes[0].plot(y_pred, label="Predicción", color="tomato", linewidth=1.2, alpha=0.8)
    axes[0].set_title(f"Retorno QQQ — horizonte {horizon}", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Retorno (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    errors = y_true - y_pred
    axes[1].hist(errors, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Distribución de errores de predicción", fontsize=13)
    axes[1].set_xlabel("Error (real − predicho %)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_generated_scenarios(
    real_returns: np.ndarray,
    fake_returns: np.ndarray,
    n_scenarios: int = 10,
    save_path: str | None = None,
) -> None:
    """
    Visualiza trayectorias reales vs generadas lado a lado.

    Args:
        real_returns: (n_samples, 20)
        fake_returns: (n_samples, 20)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for i in range(min(n_scenarios, len(real_returns))):
        axes[0].plot(real_returns[i], alpha=0.4, color="navy", linewidth=0.8)
    axes[0].set_title("Trayectorias REALES (20 días)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Días")
    axes[0].set_ylabel("Retorno (%)")
    axes[0].grid(True, alpha=0.3)

    for i in range(min(n_scenarios, len(fake_returns))):
        axes[1].plot(fake_returns[i], alpha=0.4, color="tomato", linewidth=0.8)
    axes[1].set_title("Trayectorias GENERADAS — TimeGAN (20 días)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Días")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Comparación distribucional: Real vs Sintético", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_training_history(history: dict, save_path: str | None = None) -> None:
    keys = [k for k in history if "val_" not in k and k.replace("train_", "val_") in history]
    n = len(keys)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, k in zip(axes, keys):
        metric = k.replace("train_", "")
        ax.plot(history[k], label="Train")
        ax.plot(history.get(f"val_{metric}", []), label="Val", linestyle="--")
        ax.set_title(metric, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_cumulative_returns(
    strategy_returns: np.ndarray,
    bh_returns: np.ndarray,
    save_path: str | None = None,
) -> None:
    strat_cum = np.cumprod(1 + strategy_returns / 100.0) - 1
    bh_cum    = np.cumprod(1 + bh_returns / 100.0) - 1

    plt.figure(figsize=(12, 5))
    plt.plot(strat_cum * 100, label="Estrategia Modelo", color="tomato", linewidth=1.5)
    plt.plot(bh_cum * 100,    label="Buy & Hold QQQ",    color="navy",   linewidth=1.5)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Retorno Acumulado: Estrategia vs Buy & Hold", fontsize=13, fontweight="bold")
    plt.xlabel("Días de trading")
    plt.ylabel("Retorno acumulado (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
