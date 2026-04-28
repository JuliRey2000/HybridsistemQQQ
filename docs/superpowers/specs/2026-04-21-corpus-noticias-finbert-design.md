# Diseño: Pipeline de Corpus de Noticias para FinBERT

**Fecha:** 2026-04-21  
**Proyecto:** Sistema Híbrido LSTM+FinBERT para predicción de dirección QQQ  
**Alcance:** Adquisición, normalización y cómputo de embeddings de sentimiento (2015–2024)

---

## Objetivo

Producir `data/processed/finbert_embeddings.csv` — un embedding diario de sentimiento financiero (dim 768) para cada día hábil del mercado entre 2015-01-01 y 2024-12-31 — que alimente directamente el `FinBERTSentimentLoader` ya implementado en `src/data_pipeline.py` sin modificar el código existente.

---

## Fuentes de Datos

| Fuente | Cobertura | Costo | Uso |
|--------|-----------|-------|-----|
| FNSPID (Kaggle) | 2009–2023 | Gratis | Base principal |
| Tiingo News API | 2024 | ~$10/mes (1 mes) | Completar 2024 |

Tipo de noticias: **financieras generales** (mercado amplio, Fed, macro) — mayor volumen diario que noticias específicas de QQQ/Nasdaq.

---

## Arquitectura

```
FNSPID (Kaggle)  ─┐
                   ├─► Normalización ─► Merge + Dedup ─► FinBERT Embeddings ─► CSV final
Tiingo 2024       ─┘
```

Cada etapa escribe un archivo en disco. El pipeline es **reanudable**: si se interrumpe en cualquier punto, se retoma desde el último archivo producido.

---

## Archivos del Pipeline

| Archivo | Descripción |
|---------|-------------|
| `data/raw/fnspid_news.csv` | FNSPID normalizado, filtrado 2015–2023 |
| `data/raw/tiingo_2024.csv` | Tiingo normalizado, enero–diciembre 2024 |
| `data/interim/corpus_merged.csv` | Merged + deduplicado, esquema `[date, headline, body]` |
| `data/processed/finbert_embeddings.csv` | Embeddings finales `[date, emb_0..emb_767]` |

---

## Scripts a Crear

### `scripts/download_fnspid.py`

1. Descarga vía `kaggle datasets download` (requiere `~/.kaggle/kaggle.json`)
2. Lee CSV principal de FNSPID
3. Normaliza al esquema `[date, headline, body]` — `body` vacío si no existe en la fuente
4. Filtra a rango 2015-01-01 → 2023-12-31
5. Guarda `data/raw/fnspid_news.csv`
6. Imprime reporte: total noticias, rango de fechas, % de `body` vacíos

### `scripts/download_tiingo.py`

1. Lee `TIINGO_API_KEY` desde variable de entorno (nunca hardcodeada)
2. Descarga en ventanas mensuales (enero–diciembre 2024) para respetar rate limit (~10k req/día)
3. Normaliza al esquema `[date, headline, body]`
4. Guarda `data/raw/tiingo_2024.csv`
5. Imprime reporte: noticias por mes, fechas con 0 noticias

### `scripts/build_corpus.py`

1. Carga `fnspid_news.csv` + `tiingo_2024.csv`
2. Concatena y elimina duplicados por `(date, headline)`
3. Ordena por fecha
4. Guarda `data/interim/corpus_merged.csv`
5. Imprime reporte: duplicados eliminados, cobertura diaria (% días con ≥1 noticia)

### `scripts/compute_embeddings.py`

1. Carga `corpus_merged.csv`
2. Para cada día:
   - Agrupa todas las noticias del día
   - Concatena `headline + " " + body` por noticia (truncado a 512 tokens vía tokenizador, no caracteres)
   - Extrae embedding `[CLS]` con `yiyanghkust/finbert-tone`
   - Promedia embeddings de todas las noticias del día → 1 vector (768 dim)
3. Días sin noticias: **forward-fill** del día hábil anterior (no ceros). Aplica solo a días de mercado abierto sin cobertura en el corpus — los fines de semana y festivos no aparecen en el índice de salida porque el pipeline los filtra al alinear con `price_df`.
4. Checkpoint cada 200 días → reanudable si se interrumpe
5. Batch size: 32 (CPU) / 64 (GPU Colab T4)
6. Guarda `data/processed/finbert_embeddings.csv`
7. Imprime reporte: días con embedding real vs. forward-filled, norma media de vectores

---

## Decisiones de Diseño (Mejores Prácticas)

| Decisión | Elección | Justificación |
|----------|----------|---------------|
| Días sin noticias | Forward-fill | El sentimiento persiste; ceros crean artefacto artificial |
| Truncado de texto | 512 tokens (tokenizador) | FinBERT usa tokens, no caracteres |
| Agregación diaria | Promedio simple | Estándar en literatura (Araci 2019, Liu et al. 2021) |
| Checkpoint | Cada 200 días | ~12 puntos de recuperación en 2.500 días hábiles |
| Batch size | 32 CPU / 64 GPU | Límite seguro para FinBERT sin OOM en Colab T4 |
| API key | Variable de entorno | Nunca hardcodeada en código |

---

## Integración con Pipeline Existente

Sin cambios a `src/data_pipeline.py`. Uso:

```python
pipeline = DataPipeline(
    ticker="QQQ",
    start_date="2015-01-01",
    end_date="2024-12-31",
    sentiment_path="data/processed/finbert_embeddings.csv"
)
data = pipeline.run()
```

---

## Verificación de Calidad

### Reportes automáticos (al final de cada script)
- Cobertura de fechas (% días con ≥1 noticia antes de forward-fill)
- Norma media de embeddings (norma ~0 indica problema de alineación)
- Duplicados eliminados en merge

### Test crítico post-embeddings
Verificar que el embedding del crash COVID (2020-03-16) tiene norma alta y dirección negativa. Si ese día tiene sentimiento neutro o positivo, hay un problema de alineación de fechas.

```python
# Sanity check — el umbral 1.0 es orientativo, no estricto
emb_covid = df.loc["2020-03-16"].values
assert np.linalg.norm(emb_covid) > 1.0, "Embedding COVID anormalmente bajo — revisar alineación de fechas"
```

---

## Dependencias

```
kaggle
tiingo  # o requests directo a la API REST
transformers>=4.30
torch
pandas
numpy
tqdm
```

---

## Orden de Ejecución

```bash
# 1. Configurar credenciales
export TIINGO_API_KEY=tu_key_aqui

# 2. Ejecutar pipeline en orden
python scripts/download_fnspid.py
python scripts/download_tiingo.py
python scripts/build_corpus.py
python scripts/compute_embeddings.py
```
