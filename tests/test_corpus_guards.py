"""
Tests de las guardas que protegen una corrida del corpus.

Ambas nacen del mismo incidente: la corrida del 2026-08-18 se construyó sobre un
FNSPID que solo llegaba a 2020-06 y nadie lo supo hasta después de gastar horas
de FinBERT. La cobertura corta debe detener el pipeline, y un checkpoint de
embeddings calculado sobre otro corpus no debe reutilizarse jamás.
"""
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _stub_ausentes(*nombres: str) -> list[str]:
    """
    Instala módulos vacíos solo para poder importar los scripts.

    compute_embeddings importa torch y download_fnspid importa requests, pero lo
    que aquí se verifica es pura pandas. Los stubs se retiran justo después del
    import: si quedaran en sys.modules, el código que hace `try: import torch /
    except ImportError` creería tenerlo y fallaría más adelante.
    """
    instalados = []
    for nombre in nombres:
        try:
            __import__(nombre)
        except ImportError:
            sys.modules[nombre] = types.ModuleType(nombre)
            instalados.append(nombre)
    return instalados


_stubs = _stub_ausentes("torch", "requests")
import compute_embeddings as ce
import download_fnspid as fn
import run_corpus as rc
for _nombre in _stubs:
    del sys.modules[_nombre]

DIM = 768


def _corpus(tmp_path: Path, n_filas: int) -> Path:
    """Escribe un corpus_merged.csv de juguete con `n_filas` titulares."""
    path = tmp_path / "corpus_merged.csv"
    df = pd.DataFrame({
        "date": pd.date_range("2015-01-02", periods=n_filas, freq="B"),
        "headline": [f"titular {i}" for i in range(n_filas)],
        "body": [""] * n_filas,
    })
    df.to_csv(path, index=False)
    return path


def _progreso() -> dict:
    return {"2015-01-02": np.arange(DIM, dtype=np.float32)}


# ── Checkpoint de embeddings ──────────────────────────────────────────────────

def test_sin_checkpoint_previo_el_progreso_empieza_vacio(tmp_path):
    corpus = _corpus(tmp_path, 3)
    assert ce.load_checkpoint(checkpoint_dir=tmp_path / "ckpt", corpus_path=corpus) == {}


def test_el_checkpoint_se_reutiliza_si_el_corpus_no_cambio(tmp_path):
    corpus  = _corpus(tmp_path, 3)
    ckpt    = tmp_path / "ckpt"

    ce.save_checkpoint(_progreso(), checkpoint_dir=ckpt, corpus_path=corpus)
    recuperado = ce.load_checkpoint(checkpoint_dir=ckpt, corpus_path=corpus)

    assert list(recuperado) == ["2015-01-02"]
    np.testing.assert_allclose(recuperado["2015-01-02"], np.arange(DIM, dtype=np.float32))


def test_el_checkpoint_se_descarta_si_el_corpus_se_reconstruyo(tmp_path):
    """El caso del 2026-08-18: el corpus viejo llegaba a 2020, el nuevo a 2023."""
    corpus = _corpus(tmp_path, 3)
    ckpt   = tmp_path / "ckpt"
    ce.save_checkpoint(_progreso(), checkpoint_dir=ckpt, corpus_path=corpus)

    _corpus(tmp_path, 40)   # se reconstruye el corpus con otra fuente

    assert ce.load_checkpoint(checkpoint_dir=ckpt, corpus_path=corpus) == {}


def test_un_checkpoint_sin_huella_se_descarta(tmp_path):
    """Checkpoints de versiones anteriores del script no traen huella: no fiarse."""
    corpus = _corpus(tmp_path, 3)
    ckpt   = tmp_path / "ckpt"
    ce.save_checkpoint(_progreso(), checkpoint_dir=ckpt, corpus_path=corpus)
    (ckpt / "corpus_fingerprint.txt").unlink()

    assert ce.load_checkpoint(checkpoint_dir=ckpt, corpus_path=corpus) == {}


# ── Guarda de cobertura de FNSPID ─────────────────────────────────────────────

def test_la_cobertura_completa_no_levanta_nada():
    fn.check_coverage(pd.Timestamp("2023-12-29"))


def test_la_cobertura_corta_detiene_el_pipeline():
    with pytest.raises(fn.CoverageError) as exc:
        fn.check_coverage(pd.Timestamp("2020-06-11"))

    mensaje = str(exc.value)
    assert "2020-06-11" in mensaje      # dónde acaban los datos
    assert "2023-12-31" in mensaje      # dónde deberían acabar


def test_un_hueco_de_semanas_no_detiene_el_pipeline():
    """Los últimos días de diciembre pueden faltar sin que importe."""
    fn.check_coverage(pd.Timestamp("2023-11-30"))


def test_el_override_explicito_permite_seguir_con_cobertura_corta(monkeypatch):
    monkeypatch.setenv("FNSPID_ALLOW_SHORT", "1")
    fn.check_coverage(pd.Timestamp("2020-06-11"))


# ── Cableado de la guarda dentro de main() ────────────────────────────────────

def _stats_hasta(max_date: str) -> dict:
    """Estadísticas mínimas para que print_report pueda imprimir el reporte."""
    fin = pd.Timestamp(max_date)
    return {
        "raw_rows": 10, "kept": 5, "dupes": 0,
        "by_year": Counter({fin.year: 5}),
        "days": {fin},
        "min_date": pd.Timestamp("2015-01-02"), "max_date": fin,
        "body_nonempty": 0,
    }


def _simular_descarga(monkeypatch, tmp_path: Path, max_date: str) -> Path:
    """Sustituye la descarga real por un temporal ya escrito."""
    destino = tmp_path / "fnspid_news.csv"
    monkeypatch.setattr(fn, "OUTPUT_CSV", destino)
    monkeypatch.setattr(fn, "DATA_RAW_PATH", tmp_path)

    def _fake(url, tmp):
        Path(tmp).write_text("date,headline,body\n", encoding="utf-8")
        return _stats_hasta(max_date)

    monkeypatch.setattr(fn, "stream_and_normalize", _fake)
    return destino


def test_una_descarga_corta_no_deja_el_csv_en_su_sitio(monkeypatch, tmp_path):
    """
    La trampa del 2026-08-18: el CSV mutilado quedó en disco y run_corpus.py lo
    dio por bueno en cada reintento ([SKIP] output ya existe).
    """
    destino = _simular_descarga(monkeypatch, tmp_path, "2020-06-11")

    assert fn.main() == 1
    assert not destino.exists()
    assert not destino.with_suffix(".csv.part").exists()


def test_una_descarga_completa_si_deja_el_csv_en_su_sitio(monkeypatch, tmp_path):
    destino = _simular_descarga(monkeypatch, tmp_path, "2023-12-29")

    assert fn.main() == 0
    assert destino.exists()


# ── Revalidación de un fnspid_news.csv que ya está en disco ───────────────────

def _fnspid_en_disco(tmp_path: Path, hasta: str) -> Path:
    """Deja un fnspid_news.csv ya descargado, con fechas en orden arbitrario."""
    destino = tmp_path / "fnspid_news.csv"
    fechas = [pd.Timestamp(hasta), pd.Timestamp("2015-01-02"), pd.Timestamp("2018-05-04")]
    pd.DataFrame({
        "date": fechas,
        "headline": ["a", "b", "c"],
        "body": ["", "", ""],
    }).to_csv(destino, index=False)
    return destino


def test_un_csv_ya_descargado_con_cobertura_corta_no_se_da_por_bueno(monkeypatch, tmp_path):
    """
    El fallo original: `[SKIP] output ya existe` daba por bueno el CSV que
    llegaba a 2020 y el pipeline seguía hasta gastar horas de FinBERT.
    """
    destino = _fnspid_en_disco(tmp_path, "2020-06-11")
    monkeypatch.setattr(fn, "OUTPUT_CSV", destino)

    assert fn.main() == 1


def test_un_csv_ya_descargado_y_completo_se_reutiliza(monkeypatch, tmp_path):
    destino = _fnspid_en_disco(tmp_path, "2023-12-29")
    monkeypatch.setattr(fn, "OUTPUT_CSV", destino)

    assert fn.main() == 0
    assert destino.exists()


# ── El orquestador debe dejar que FNSPID se revalide ──────────────────────────

def _espiar_subprocess(monkeypatch) -> list:
    """Registra las órdenes lanzadas por run_step sin ejecutar nada."""
    ejecutados = []

    def _fake_run(cmd, cwd=None):
        ejecutados.append(cmd)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(rc.subprocess, "run", _fake_run)
    return ejecutados


def test_un_paso_normal_se_salta_si_su_output_existe(monkeypatch, tmp_path):
    ejecutados = _espiar_subprocess(monkeypatch)
    salida = tmp_path / "corpus_merged.csv"
    salida.write_text("date,headline,body\n", encoding="utf-8")

    assert rc.run_step({"name": "X", "script": tmp_path / "x.py", "output": salida})
    assert ejecutados == []


def test_un_paso_revalidable_se_ejecuta_aunque_su_output_exista(monkeypatch, tmp_path):
    ejecutados = _espiar_subprocess(monkeypatch)
    salida = tmp_path / "fnspid_news.csv"
    salida.write_text("date,headline,body\n", encoding="utf-8")

    assert rc.run_step({
        "name": "X", "script": tmp_path / "x.py", "output": salida, "always_run": True,
    })
    assert len(ejecutados) == 1


def test_fnspid_esta_marcado_como_revalidable():
    """
    Si el paso se saltara, la guarda de cobertura de download_fnspid.py nunca
    llegaría a correr sobre el archivo que ya está en disco.
    """
    paso = next(p for p in rc.STEPS if "FNSPID" in p["name"])
    assert paso.get("always_run") is True
