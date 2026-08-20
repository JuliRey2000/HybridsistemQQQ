"""
Tests de la alineación entre los embeddings de FinBERT y los días de mercado.

`create_sequences` empareja el sentimiento por igualdad exacta de fecha y, si no
encuentra el día, mete un vector de ceros **en silencio**. Con eso, un desajuste
de fechas no produce ningún error: la rama de sentimiento se queda vacía y el
modelo entrena como si fuera solo-precio. El chequeo del notebook tampoco lo
detecta, porque solo mira si existe *algún* valor distinto de cero.

Es la tercera vez que este proyecto se topa con un fallo que no falla. Aquí se
cierra: si se pasaron embeddings, tienen que alinear.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _stub_ausentes(*nombres: str) -> list[str]:
    """Módulos vacíos solo para importar data_pipeline; se retiran tras el import."""
    instalados = []
    for nombre in nombres:
        try:
            __import__(nombre)
        except ImportError:
            sys.modules[nombre] = types.ModuleType(nombre)
            instalados.append(nombre)
    return instalados


_stubs = _stub_ausentes("yfinance", "ta")
from data_pipeline import create_sequences
for _nombre in _stubs:
    del sys.modules[_nombre]

DIM      = 8      # basta para el test; el real es 768
LOOKBACK = 3
N_DIAS   = 40


def _price_df() -> pd.DataFrame:
    dias = pd.date_range("2015-01-02", periods=N_DIAS, freq="B")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "RSI":          rng.normal(size=N_DIAS),
            "VIX_Close":    rng.normal(size=N_DIAS),
            "Daily_Return": rng.normal(size=N_DIAS),
        },
        index=dias,
    )


def _sentiment_df(dias: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        np.ones((len(dias), DIM), dtype=np.float32),
        index=dias,
        columns=[f"emb_{i}" for i in range(DIM)],
    )


def _secuencias(price_df, sentiment_df):
    return create_sequences(price_df, sentiment_df, lookback=LOOKBACK)


def test_con_fechas_alineadas_ningun_dia_queda_en_ceros():
    price_df = _price_df()
    datos = _secuencias(price_df, _sentiment_df(price_df.index))

    en_cero = (datos["sentiments"] == 0).all(axis=1).sum()
    assert en_cero == 0


def test_un_desajuste_de_fechas_no_puede_pasar_desapercibido():
    """El fallo silencioso: los embeddings existen pero no cuadran con los precios."""
    price_df = _price_df()
    desplazado = _sentiment_df(price_df.index + pd.Timedelta("1D"))

    with pytest.raises(RuntimeError) as exc:
        _secuencias(price_df, desplazado)

    assert "sentimiento" in str(exc.value).lower()


def test_una_cobertura_parcial_tambien_levanta():
    """Media serie alineada sigue siendo un bug, no una propiedad de los datos."""
    price_df = _price_df()
    mitad = _sentiment_df(price_df.index[: N_DIAS // 2])

    with pytest.raises(RuntimeError):
        _secuencias(price_df, mitad)


def test_faltar_un_dia_suelto_no_levanta():
    """Un hueco aislado es tolerable; lo que no lo es es un desajuste sistemático."""
    price_df = _price_df()
    casi_todo = _sentiment_df(price_df.index.drop(price_df.index[5]))

    datos = _secuencias(price_df, casi_todo)
    assert len(datos["sentiments"]) > 0


def test_sin_embeddings_los_ceros_son_legitimos():
    """El placeholder de ceros es el modo esperado mientras no exista el corpus."""
    datos = _secuencias(_price_df(), None)

    assert (datos["sentiments"] == 0).all()
