"""
Data Pipeline: Recolección y preprocesamiento de datos para predicción QQQ.

Cubre el periodo 2015-2024 conforme a la delimitación de la tesis.
El crash del COVID-19 (marzo 2020) se retiene sin modificación alguna.

Fases:
  1. Descargar OHLCV de QQQ (Yahoo Finance)
  2. Calcular retornos diarios y garantizar estacionariedad
  3. Añadir indicadores técnicos (RSI, MACD, Bollinger, ATR, SMA)
  4. Extraer embeddings de sentimiento con FinBERT (o cargarlos desde disco)
  5. Empaquetar en ventanas deslizantes (sliding windows)
  6. Guardar dataset para entrenamiento

Estándar: CRISP-ML(Q) — reproducibilidad y trazabilidad de cada paso.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ============================================================================
# PASO 1 & 2: PRECIOS + RETORNOS
# ============================================================================

class PriceDataLoader:
    """Descarga OHLCV y calcula retornos logarítmicos para estacionariedad."""

    def __init__(
        self,
        ticker: str = "QQQ",
        start_date: str = "2015-01-01",
        end_date: str = "2024-12-31",
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def load(self) -> pd.DataFrame:
        """
        Descarga y prepara el dataframe de precios.

        Returns:
            df con columnas OHLCV + VIX_Close + Daily_Return (log-return %)
            El índice es DatetimeIndex (días de mercado abierto).
        """
        logger.info(f"Descargando {self.ticker} [{self.start_date} → {self.end_date}]")
        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise RuntimeError(f"yfinance no devolvió datos para {self.ticker}")

        df = df.dropna()

        # Aplanar MultiIndex de columnas si existe
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Retorno logarítmico diario (más estacionario que precio)
        # r_t = 100 * ln(P_t / P_{t-1})
        df["Daily_Return"] = 100.0 * np.log(df["Close"] / df["Close"].shift(1))
        df = df.dropna()

        # VIX: índice de volatilidad implícita (proxy del miedo del mercado)
        # Inner join para retener solo días donde ambas series tienen datos
        logger.info("Descargando ^VIX (índice de volatilidad del mercado)...")
        vix = yf.download("^VIX", start=self.start_date, end=self.end_date,
                          progress=False, auto_adjust=True)
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix_close = vix[["Close"]].rename(columns={"Close": "VIX_Close"}).dropna()
            df = df.join(vix_close, how="inner")
            df = df.dropna()
            logger.info(f"VIX añadido. Días tras inner join: {len(df)}")
        else:
            logger.warning("No se pudo descargar VIX — se continuará sin este feature.")

        logger.info(
            f"Cargados {len(df)} días | "
            f"Retorno medio: {df['Daily_Return'].mean():.4f}% | "
            f"Retorno std: {df['Daily_Return'].std():.4f}%"
        )

        # Verificación COVID: el crash de marzo 2020 debe estar presente
        covid_mask = (df.index >= "2020-03-01") & (df.index <= "2020-04-30")
        covid_min = df.loc[covid_mask, "Daily_Return"].min() if covid_mask.any() else None
        if covid_min is not None:
            logger.info(
                f"[COVID check] Retorno mínimo en mar-abr 2020: {covid_min:.2f}% "
                f"— Datos de crisis retenidos correctamente."
            )

        return df


# ============================================================================
# PASO 3: INDICADORES TÉCNICOS
# ============================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade indicadores técnicos al dataframe de precios.

    Indicadores añadidos:
      RSI_14       : Momentum
      MACD / Signal/ Diff : Cruces de medias móviles
      BB_Pct       : Posición dentro de las bandas de Bollinger (0-1)
      ATR_14       : Volatilidad realizada (Average True Range)
      SMA_20 / SMA_50: Tendencia de corto y mediano plazo
      Vol_Change   : Cambio porcentual en volumen (señal de actividad)

    Args:
        df: DataFrame con columnas OHLCV

    Returns:
        df ampliado con indicadores (sin filas NaN)
    """
    df = df.copy()

    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)

    macd = ta.trend.MACD(df["Close"])
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"]   = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Pct"]  = bb.bollinger_pband()   # porcentaje dentro de las bandas

    df["ATR_14"]  = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=14
    )

    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)

    df["Vol_Change"] = df["Volume"].pct_change() * 100

    df = df.dropna()
    logger.info(
        f"Indicadores técnicos calculados. Columnas: {df.columns.tolist()} | "
        f"Registros finales: {len(df)}"
    )
    return df


# ============================================================================
# PASO 4: SENTIMIENTO (FinBERT)
# ============================================================================

class FinBERTSentimentLoader:
    """
    Carga o computa embeddings de sentimiento usando FinBERT.

    Dos modos:
      - Si existe `sentiment_path` en disco: carga directo (rápido, reproducible)
      - Si no existe: computa desde corpus de noticias (requiere HuggingFace)

    El sentimiento asíncrono (múltiples noticias por día) se consolida en
    un promedio ponderado diario alineado con el cierre del mercado.

    El embedding resultante tiene dimensión 768 (capa [CLS] de FinBERT).
    """

    FINBERT_MODEL = "yiyanghkust/finbert-tone"

    def __init__(
        self,
        sentiment_path: Optional[str] = None,
        news_csv_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            sentiment_path: Ruta a CSV con columnas [date, emb_0, ..., emb_767]
                            Si existe, se usa directamente sin computar.
            news_csv_path:  Ruta a CSV con columnas [date, headline, body]
                            Se usa para computar embeddings desde cero.
            device:         'cpu' o 'cuda'
        """
        self.sentiment_path = sentiment_path
        self.news_csv_path = news_csv_path
        self.device = device

    def load_precomputed(self) -> pd.DataFrame:
        """Carga embeddings pre-computados desde disco."""
        df = pd.read_csv(self.sentiment_path, index_col=0, parse_dates=True)
        logger.info(f"Embeddings FinBERT cargados desde {self.sentiment_path} | shape: {df.shape}")
        return df

    def compute_from_news(self) -> pd.DataFrame:
        """
        Computa embeddings FinBERT desde corpus de noticias.

        Requiere: pip install transformers torch

        El texto de cada noticia pasa por FinBERT y se extrae el embedding
        del token [CLS] (representación global del documento).
        Múltiples noticias del mismo día se promedian.

        Returns:
            DataFrame con index=date, columnas emb_0..emb_767
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            raise ImportError("Instala: pip install transformers")

        if not self.news_csv_path or not Path(self.news_csv_path).exists():
            raise FileNotFoundError(
                f"No se encontró corpus de noticias en: {self.news_csv_path}"
            )

        news_df = pd.read_csv(self.news_csv_path, parse_dates=["date"])
        logger.info(f"Corpus cargado: {len(news_df)} noticias")

        tokenizer = AutoTokenizer.from_pretrained(self.FINBERT_MODEL)
        model = AutoModel.from_pretrained(self.FINBERT_MODEL).to(self.device)
        model.eval()

        records = []
        for date, group in news_df.groupby("date"):
            day_embeddings = []
            for _, row in group.iterrows():
                text = str(row.get("headline", "")) + " " + str(row.get("body", ""))
                text = text[:512]  # FinBERT max tokens

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    out = model(**inputs)
                cls_emb = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                day_embeddings.append(cls_emb)

            avg_emb = np.mean(day_embeddings, axis=0)  # promedio ponderado diario
            records.append({"date": date, **{f"emb_{i}": v for i, v in enumerate(avg_emb)}})

        result = pd.DataFrame(records).set_index("date")
        logger.info(f"Embeddings FinBERT computados: {result.shape}")
        return result

    def load(self) -> pd.DataFrame:
        """Punto de entrada: elige el modo según disponibilidad."""
        if self.sentiment_path and Path(self.sentiment_path).exists():
            return self.load_precomputed()
        elif self.news_csv_path:
            return self.compute_from_news()
        else:
            logger.warning(
                "No se proporcionó fuente de sentimiento. "
                "Se usará tensor de ceros como placeholder."
            )
            return None


# ============================================================================
# PASO 5: EMPAQUETAR EN VENTANAS DESLIZANTES
# ============================================================================

def create_sequences(
    price_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame],
    lookback: int = 30,
    horizon_t1: int = 1,
    horizon_t5: int = 5,
    price_feature_cols: Optional[list] = None,
) -> dict:
    """
    Crea ventanas deslizantes (sliding windows) para entrenamiento.

    Cada muestra contiene:
      - price_seq  : últimos `lookback` días de features técnicos
      - sentiment  : embedding FinBERT del día actual (o cero si no hay)
      - y_t1       : retorno en t+1
      - y_t5       : retorno en t+5
      - dates      : fecha del día de predicción

    El orden temporal se preserva estrictamente. No hay aleatorización.

    Args:
        price_df      : DataFrame con features técnicos + Daily_Return
        sentiment_df  : DataFrame con embeddings FinBERT (index=date) o None
        lookback      : días de historia (default: 30)
        horizon_t1    : días al futuro para y_t1 (default: 1)
        horizon_t5    : días al futuro para y_t5 (default: 5)
        price_feature_cols: columnas a usar como features (si None, auto-selecciona)

    Returns:
        dict con arrays numpy: price_seqs, sentiments, y_t1, y_t5, dates
    """
    if price_feature_cols is None:
        price_feature_cols = [
            c for c in price_df.columns
            if c not in ["Daily_Return", "Close", "Open", "High", "Low", "Volume"]
        ]

    logger.info(f"Features de precio: {price_feature_cols}")
    logger.info(f"Creando secuencias lookback={lookback}, t+1, t+5")

    sentiment_dim = sentiment_df.shape[1] if sentiment_df is not None else 768
    max_horizon = max(horizon_t1, horizon_t5)

    price_seqs, sentiments, y_t1s, y_t5s, dates = [], [], [], [], []

    n = len(price_df)
    for i in range(lookback, n - max_horizon):
        # Ventana de precios: [i-lookback, i)
        seq = price_df[price_feature_cols].iloc[i - lookback:i].values
        price_seqs.append(seq)

        # Sentimiento del día i (si existe)
        if sentiment_df is not None:
            day = price_df.index[i]
            if day in sentiment_df.index:
                sent = sentiment_df.loc[day].values.astype(np.float32)
            else:
                sent = np.zeros(sentiment_dim, dtype=np.float32)
        else:
            sent = np.zeros(sentiment_dim, dtype=np.float32)
        sentiments.append(sent)

        # Targets
        y_t1s.append(price_df["Daily_Return"].iloc[i + horizon_t1])
        y_t5s.append(price_df["Daily_Return"].iloc[i + horizon_t5])
        dates.append(price_df.index[i])

    return {
        "price_seqs":  np.array(price_seqs,  dtype=np.float32),
        "sentiments":  np.array(sentiments,  dtype=np.float32),
        "y_t1":        np.array(y_t1s,       dtype=np.float32),
        "y_t5":        np.array(y_t5s,       dtype=np.float32),
        "dates":       np.array(dates),
    }


# ============================================================================
# PASO 6: NORMALIZACIÓN
# ============================================================================

def fit_scalers(
    price_df: pd.DataFrame,
    train_end_idx: int,
    feature_cols: list,
) -> StandardScaler:
    """
    Ajusta el scaler SOLO con datos de entrenamiento para evitar data leakage.

    Args:
        price_df    : DataFrame completo
        train_end_idx: Índice donde termina el conjunto de entrenamiento
        feature_cols : Columnas a normalizar

    Returns:
        scaler ajustado (aplicar luego a val/test con .transform())
    """
    train_features = price_df[feature_cols].iloc[:train_end_idx]
    scaler = StandardScaler()
    scaler.fit(train_features)
    logger.info(f"Scaler ajustado sobre {train_end_idx} días de entrenamiento.")
    return scaler


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

class DataPipeline:
    """
    Orquesta todo el ciclo CRISP-ML(Q) de ingeniería de datos.

    Uso típico:
        pipeline = DataPipeline()
        data = pipeline.run()
        # data['price_seqs'], data['sentiments'], data['y_t1'], data['y_t5']
    """

    def __init__(
        self,
        ticker: str = "QQQ",
        start_date: str = "2015-01-01",
        end_date: str = "2024-12-31",
        lookback: int = 30,
        sentiment_path: Optional[str] = None,
        news_csv_path: Optional[str] = None,
        save_dir: str = "data/processed",
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.sentiment_path = sentiment_path
        self.news_csv_path = news_csv_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        logger.info("=" * 65)
        logger.info("INICIANDO PIPELINE — CRISP-ML(Q)")
        logger.info("=" * 65)

        # 1-2. Precios + retornos
        loader = PriceDataLoader(self.ticker, self.start_date, self.end_date)
        price_df = loader.load()

        # 3. Indicadores técnicos
        price_df = add_technical_indicators(price_df)

        # 4. Sentimiento FinBERT
        sent_loader = FinBERTSentimentLoader(
            sentiment_path=self.sentiment_path,
            news_csv_path=self.news_csv_path,
        )
        sentiment_df = sent_loader.load()

        # 5. Ventanas deslizantes
        data = create_sequences(
            price_df=price_df,
            sentiment_df=sentiment_df,
            lookback=self.lookback,
        )

        # 6. Guardar
        for key, arr in data.items():
            if isinstance(arr, np.ndarray):
                np.save(self.save_dir / f"{key}.npy", arr)
        price_df.to_csv(self.save_dir / "price_df.csv")

        logger.info("=" * 65)
        logger.info("PIPELINE COMPLETADO")
        logger.info(f"  price_seqs : {data['price_seqs'].shape}")
        logger.info(f"  sentiments : {data['sentiments'].shape}")
        logger.info(f"  y_t1       : {data['y_t1'].shape}")
        logger.info(f"  y_t5       : {data['y_t5'].shape}")
        logger.info(f"  Guardado en: {self.save_dir}")
        logger.info("=" * 65)

        return data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    pipeline = DataPipeline(
        ticker="QQQ",
        start_date="2015-01-01",
        end_date="2024-12-31",
        lookback=30,
        # sentiment_path="data/processed/finbert_embeddings.csv",  # si ya existe
        # news_csv_path="data/raw/news.csv",                       # si quieres computar
    )
    data = pipeline.run()
    print(f"\nForma price_seqs : {data['price_seqs'].shape}")
    print(f"Forma sentiments : {data['sentiments'].shape}")
    print(f"Forma y_t1       : {data['y_t1'].shape}")
