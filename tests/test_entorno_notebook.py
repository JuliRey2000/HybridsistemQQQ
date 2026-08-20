"""
Tests del entorno que instala el notebook.

Origen: el 2026-08-20 la corrida murió al cargar el tokenizador de FinBERT.
`yiyanghkust/finbert-tone` publica solo `vocab.txt` (sin `tokenizer.json` ni
`tokenizer_config.json`), así que depende de que transformers construya el
tokenizador *lento* desde el vocabulario y lo convierta. transformers 5 eliminó
esa ruta y el modelo dejó de cargar.

`requirements.txt` acotaba la versión correctamente, pero la celda de
dependencias del notebook hacía `pip install --upgrade transformers`, sin pin,
y se llevaba por delante esa restricción. El entorno declarado y el instalado
no eran el mismo — eso es lo que se verifica aquí.
"""
import json
import re
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

REPO     = Path(__file__).parent.parent
NOTEBOOK = REPO / "notebooks" / "QQQ_Hibrido_Completo.ipynb"
REQS     = REPO / "requirements.txt"

# Versión que rompe el tokenizador de finbert-tone.
PRIMERA_V5 = "5.0.0"


def _celda_de_dependencias() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for celda in nb["cells"]:
        fuente = "".join(celda["source"])
        if celda["cell_type"] == "code" and "pip install" in fuente:
            return fuente
    raise AssertionError("El notebook no tiene celda de dependencias.")


def _spec_de(texto: str, paquete: str) -> SpecifierSet:
    """
    Extrae la restricción de versión de un paquete en texto de pip.

    Descarta las líneas de comentario: el propio comentario que explica el pin
    menciona el paquete, y sin filtrarlas la primera coincidencia sería esa —
    una restricción vacía que dejaría pasar cualquier versión.
    """
    activo = "".join(
        linea for linea in texto.splitlines(keepends=True)
        if not linea.strip().startswith("#")
    )
    limpio = activo.replace(" ", "").replace('"', "").replace("'", "")

    match = re.search(rf"{paquete}((?:[<>=!~]=?[\d.]+,?)*)", limpio)
    if match is None:
        raise AssertionError(f"'{paquete}' no aparece en el texto.")
    return SpecifierSet(match.group(1))


def test_el_notebook_acota_transformers_por_debajo_de_v5():
    """Sin pin, Colab instala la v5 y finbert-tone deja de cargar."""
    spec = _spec_de(_celda_de_dependencias(), "transformers")

    assert PRIMERA_V5 not in spec, (
        f"La celda de dependencias permite transformers {PRIMERA_V5}. "
        f"Esa versión no puede cargar yiyanghkust/finbert-tone, que solo "
        f"publica vocab.txt."
    )


def test_requirements_acota_transformers_por_debajo_de_v5():
    spec = _spec_de(REQS.read_text(encoding="utf-8"), "transformers")

    assert PRIMERA_V5 not in spec


def test_el_notebook_no_contradice_a_requirements():
    """
    Las dos fuentes de verdad tienen que ser compatibles. Si divergen, el
    entorno que se prueba no es el que se declara — que es justo lo que pasó.
    """
    del_notebook = _spec_de(_celda_de_dependencias(), "transformers")
    de_requirements = _spec_de(REQS.read_text(encoding="utf-8"), "transformers")

    versiones = ["4.35.2", "4.40.0", "4.57.0", "5.0.0", "5.1.0"]
    permitidas_nb = {v for v in versiones if v in del_notebook}
    permitidas_rq = {v for v in versiones if v in de_requirements}

    assert permitidas_nb & permitidas_rq, (
        f"El notebook permite {sorted(permitidas_nb)} y requirements.txt "
        f"{sorted(permitidas_rq)}: no hay ninguna versión que satisfaga a ambos."
    )
