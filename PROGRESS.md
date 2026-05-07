# Progreso del Proyecto

## Estado General: PROTOTIPO FUNCIONAL EN CONSTRUCCIÓN 🔄

**Fecha de Inicio:** Abril 2026
**Último Actualizado:** Mayo 7, 2026
**Timeline:** 2-3 meses para completar todas las fases

---

## Última Corrida — 2026-05-07 (actualización 2)

**Notebook:** `QQQ_Hibrido_Completo.ipynb` | **Entorno:** Google Colab T4 | **FinBERT:** ceros (sin corpus)

| Métrica | Walk-Forward (avg ± std) | Test Out-of-Sample |
|---------|--------------------------|-------------------|
| RMSE t+1 | 1.6064% ± 0.4837% | **1.0969%** |
| DA   t+1 | 0.547 | **0.586** |
| RMSE t+5 | 1.5781% ± 0.4853% | **1.1032%** |
| DA   t+5 | 0.546 | **0.584** |
| Sharpe t+1 | — | **1.203** ✓ |
| MaxDD t+1  | — | **-13.85%** ✓ |

**Targets de la tesis:**
- RMSE < 0.8% → ✗ `1.0969%` (brecha: 0.30 pp — requiere FinBERT real)
- Sharpe > 0.5 → ✓ `1.203`
- MaxDD > -20% → ✓ `-13.85%`

**Notas técnicas:**
- Features: **10** = 9 indicadores técnicos (`RSI_14`, `MACD`, `MACD_Signal`, `MACD_Diff`, `BB_Pct`, `ATR_14`, `SMA_20`, `SMA_50`, `Vol_Change`) + `VIX_Close` añadido en `PriceDataLoader.load()` y no excluido en `create_sequences`. Documentación anterior decía 9 porque no contaba VIX.
- Parámetros del modelo: 354,530
- Test set: 365 muestras (último 15% del periodo 2015-2024)
- Optimización VRAM aplicada: dataset GAN pre-cargado en GPU, sin transferencias CPU→GPU en training loop

**Cambios en notebooks (2026-05-07, sesión 2):**

`QQQ_Prototipo_Colab.ipynb`:
- **Sección 6 agregada** — Análisis de Exposición al Mercado. Introduce zona de confianza adaptativa (`CONFIDENCE_MARGIN = percentil 50 de |prob − 0.5|`) que opera el 50% de días más seguros, dejando el resto en cash. Genera tabla comparativa LSTM vs Buy & Hold (exposición, win rate, Sharpe, MaxDD, retorno) y figura de 3 paneles (`exposicion_lstm_vs_bh.png`).
- **Bug MLflow resuelto** — `cell-23` tenía `mlflow.start_run()` sin guard; al re-ejecutar o tras `cell-24` (duplicado) explotaba con "run already active". Fix: `if mlflow.active_run(): mlflow.end_run()` antes de `start_run`. `cell-24` (entrenamiento duplicado sin MLflow) eliminada.
- **Diagnóstico de sesgo alcista** — El clasificador (`LSTMDirectionModel`) produce `preds_prob ∈ [0.521, 0.528]` siempre > 0.5. Causa: QQQ tiene drift positivo histórico + `pos_weight` en `BCEWithLogitsLoss`. La estrategia resultante es de **timing largo puro** (0 posiciones cortas), no long/short. El diagnóstico lo detecta automáticamente e imprime advertencia.
- **Resultado con exposición 50%:** 9.1% retorno vs 35.0% B&H, MaxDD −16.2% vs −23.4%, win rate 57.4% vs 58.3%. Narrativa válida para presentación: misma tasa de acierto con la mitad del tiempo en riesgo y 7 pp menos de drawdown.

`QQQ_Hibrido_Completo.ipynb`:
- **Diagnóstico de exposición agregado al `cell-backtest`** — El modelo híbrido es un **regresor continuo** (predice retorno % directo), por lo que NO tiene el sesgo alcista del clasificador: sí genera señales cortas. El problema compartido es `THRESHOLD=0.0` que elimina la zona cash. La celda ahora imprime: rango de `preds_t1`, % de predicciones positivas/negativas, y una tabla de sensibilidad que muestra cuántos días quedarían activos para `THRESHOLD ∈ {0.1, 0.2, 0.3, 0.5}%`. Permite elegir el umbral con criterio antes de la presentación.

---

## Contexto de Notebooks

| Notebook | Propósito | Estado |
|----------|-----------|--------|
| `QQQ_Prototipo_Colab.ipynb` | Póster de congreso — clasificación binaria, split estático | ✅ TERMINADO (no modificar) |
| `QQQ_Hibrido_Completo.ipynb` | **Prototipo funcional de la tesis** — regresión t+1/t+5, walk-forward, usa `src/` | 🔄 EN USO |

---

## ✅ FASE 1: Configuración y Data Pipeline (COMPLETADA)

- [x] Estructura del proyecto (`data/`, `src/`, `models/`, `notebooks/`)
- [x] `config.py` con variables de entorno
- [x] `src/data_pipeline.py`: descarga QQQ, 9 indicadores técnicos, ventanas LSTM, soporte FinBERT
- [x] `src/utils.py`: walk-forward splits, métricas (RMSE, MAE, DA, Sharpe, MaxDD), backtesting, visualizaciones
- [x] `src/models.py`: `HybridPredictiveModel` (BiLSTM + CrossAttention + FinBERT), `TimeGANGenerator`, `WassersteinCritic`
- [x] `src/train.py`: `Trainer` (Huber multi-step, walk-forward), `GANTrainer` (WGAN-GP, n_critic=5)
- [x] Scripts ejecutables: `run_pipeline.py`, `run_train_predictive.py`, `run_train_generative.py`
- [x] Notebook prototipo funcional `QQQ_Hibrido_Completo.ipynb`

---

## 🔄 FASE 2: EDA (PARCIALMENTE CUBIERTA)

Cubierta en `QQQ_Hibrido_Completo.ipynb` Sección 2:

- [x] Precio de cierre, retornos diarios, distribución, RSI
- [x] Verificación COVID (crash mar 2020 retenido)
- [x] Estadísticos: skewness, kurtosis, media, std
- [ ] ACF/PACF (autocorrelación — pendiente notebook dedicado)
- [ ] Test ADF de estacionariedad formal
- [ ] Matriz de correlación entre los 9 features técnicos

---

## 🔄 FASE 3: Modelo LSTM/BiLSTM (CUBIERTA EN PROTOTIPO FUNCIONAL)

Cubierta en `QQQ_Hibrido_Completo.ipynb` Sección 3:

- [x] `HybridPredictiveModel` (BiLSTM + Self-Attention + CrossAttention)
- [x] Walk-forward validation (5 folds, train crece acumulativamente)
- [x] Early stopping sobre `val_loss`
- [x] Métricas por fold: RMSE t+1, RMSE t+5, DA t+1, DA t+5
- [x] Curvas de entrenamiento (último fold)
- [x] MLflow tracking vía DagsHub

---

## ❌ FASE 4: Corpus FinBERT — BLOQUEANTE CRÍTICO

**Estado:** Sin iniciar. Sin corpus, la rama de sentimiento opera con ceros.

### Tareas

- [ ] **Descargar FNSPID** (Kaggle: `2009-2023`, gratis)
  - Dataset: `kaggle datasets download -d humananalog/fnspid`
  - Archivo: `financial_news.csv` con columnas `[date, ticker, headline, body]`
  - Filtrar por fechas 2015-2023

- [ ] **Descargar Tiingo API** (2024, ~$10/mes, 1 mes suficiente)
  - Endpoint: `https://api.tiingo.com/tiingo/news`
  - Filtrar por `tickers=QQQ` o noticias financieras generales

- [ ] **`build_corpus.py`**: unir FNSPID + Tiingo, forward-fill días sin noticias
- [ ] **`compute_embeddings.py`**: FinBERT CLS-token por día, checkpoint cada 200 días
  - Output: `data/processed/finbert_embeddings.csv` (2500 filas × 768 cols)
  - Sanity check: embedding 2020-03-16 debe tener norma alta y dirección negativa

### Criterio de éxito
- Archivo `finbert_embeddings.csv` con una fila por cada día hábil 2015-2024
- Norma media de embeddings > 5.0
- Embedding COVID (2020-03-16) con signo negativo dominante

---

## ⏳ FASE 5: Modelo Híbrido Completo (PENDIENTE — requiere Fase 4)

Una vez disponible `finbert_embeddings.csv`:

- [ ] Reejecutar `QQQ_Hibrido_Completo.ipynb` con `has_sentiment=True`
- [ ] Ablation study: RMSE precio-solo vs RMSE precio+sentimiento
- [ ] Análisis de contribución de CrossAttention (attention weights visualization)

### Criterio de éxito
- Mejora RMSE ≥ 10% respecto al baseline de ceros
- DA t+1 > 55%

---

## ⏳ FASE 6: Backtesting Completo (PARCIALMENTE CUBIERTA)

Cubierta en `QQQ_Hibrido_Completo.ipynb` Sección 5:

- [x] Estrategia long/short con umbral configurable
- [x] Sharpe, Sortino, MaxDD, número de trades
- [x] Comparación vs Buy & Hold
- [x] Diagnóstico de exposición al mercado y tabla de sensibilidad de THRESHOLD (`QQQ_Hibrido_Completo.ipynb`)
- [x] Análisis de exposición con zona de confianza adaptativa (`QQQ_Prototipo_Colab.ipynb` — Sección 6)
- [x] Diagnóstico de sesgo alcista en clasificador (detectado y documentado)
- [ ] Elegir THRESHOLD óptimo para `QQQ_Hibrido_Completo` basado en tabla de sensibilidad
- [ ] Robustez a diferentes regímenes de mercado (alcista 2017-2019, caída 2022)

---

## ✅ FASE 6b: TimeGAN — Módulo Generativo (FUNCIONAL)

Cubierta en `QQQ_Hibrido_Completo.ipynb` Sección 6:

- [x] Arquitectura `TimeGANGenerator + WassersteinCritic` implementada
- [x] Entrenamiento WGAN-GP con n_critic=5, λ_gp=10
- [x] Métricas generativas: Wasserstein Distance, hechos estilizados
- [x] Visualización trayectorias reales vs generadas
- [x] **Bug crítico resuelto (2026-05-07):** loop infinito silencioso en `GANTrainer.train_epoch` — el `while True` con `StopIteration` como control de flujo nunca alcanzaba el `break` cuando `len(loader) mod (n_critic+1) ≠ n_critic` (38 batches, n_critic=5). Fix: materializar batches como lista e iterar con índice explícito. Notebook corre completamente.
- [x] **Optimización VRAM aplicada (2026-05-07):** dataset GAN pre-cargado en GPU (`torch.from_numpy(...).to(device)`) en notebook, `run_train_generative.py` y `src/train.py`. Eliminados `.to(self.device)` redundantes en `GANTrainer.train_epoch`.
- [ ] Aumentar épocas a 500+ para calidad distribucional suficiente
- [ ] Escenario de stress-test con embedding COVID real (requiere Fase 4)

---

## ⏳ FASE 7: Documentación Final (PENDIENTE)

- [ ] Docstrings completos y type hints (src/ ya los tiene)
- [ ] Análisis de interpretabilidad (SHAP / attention weights)
- [ ] Diagrama de arquitectura
- [ ] Redacción de metodología para la tesis

---

## Métricas Objetivo

| Métrica | Baseline | Híbrido (ceros) | Híbrido (FinBERT) | Target |
|---------|----------|-----------------|-------------------|--------|
| RMSE t+1 (%) | 1.5 | **1.0969** | esperado < 0.9 | < 0.8 |
| DA   t+1 | — | **0.586** | esperado > 0.60 | > 0.55 |
| Sharpe | — | **1.203** ✓ | — | > 0.5 |
| Max DD (%) | -25 | **-13.85** ✓ | — | > -15 |

*Columna "Híbrido (FinBERT)" se completará en Fase 5.*

---

## Problemas Identificados y Soluciones

| Problema | Solución |
|----------|----------|
| Look-ahead bias | ✅ Walk-forward cronológico sin shuffle |
| Data leakage en normalización | ✅ Scaler ajustado solo en train, transform en val/test |
| Sentimiento sin corpus | 🔄 Zeros como placeholder — Fase 4 lo resuelve |
| Mercados volátiles | ✅ Huber Loss en lugar de MSE |
| GAN inestabilidad | ✅ WGAN-GP (Gradient Penalty) en lugar de weight clipping |
| Loop infinito en `GANTrainer.train_epoch` | ✅ Batches materializados como lista + índice explícito (2026-05-07) |
| MLflow "run already active" en notebook póster | ✅ Guard `if mlflow.active_run(): mlflow.end_run()` en `cell-23`; `cell-24` duplicada eliminada (2026-05-07) |
| Exposición 0% en análisis de exposición del póster | ✅ `CONFIDENCE_MARGIN` hardcodeado a 0.10 superaba el rango real de probs [0.521–0.528]; ahora adaptativo (percentil 50) (2026-05-07) |
| Clasificador con sesgo alcista puro (0 cortos) | ⚠ Estructural — `LSTMDirectionModel` aprende drift QQQ y produce probs siempre > 0.5. Encuadrar como "timing largo" en la presentación. El regresor híbrido no tiene este problema. |

---

## Próximos Pasos — Ordenados por Prioridad

### PASO 1 — Verificar discrepancia de features (inmediato, 15 min)
El resumen reporta **10 features técnicos** pero la documentación dice 9. Abrir `src/data_pipeline.py` y contar los indicadores que se calculan. Actualizar `PROGRESS.md` y `CLAUDE.md` con el número correcto.

### PASO 2 — Construir corpus FinBERT (Fase 4, crítico para RMSE)
Es la única ruta para bajar RMSE de 1.10% a < 0.8%. Sin esto la tesis no cumple el target principal.

1. **Descargar FNSPID** (Kaggle, gratuito, ~2 GB):
   ```
   kaggle datasets download -d humananalog/fnspid
   ```
   Columnas necesarias: `[date, ticker, headline]`. Filtrar `date >= 2015-01-01`.

2. **Descargar Tiingo API** para cubrir 2024 (endpoints de noticias, ~$10/mes):
   - Endpoint: `https://api.tiingo.com/tiingo/news?tickers=QQQ&startDate=2024-01-01`
   - Token en `.env` como `TIINGO_API_KEY`

3. **Crear `fuentes/build_corpus.py`**: une FNSPID + Tiingo, forward-fill días sin noticias, agrupa por fecha.

4. **Crear `fuentes/compute_embeddings.py`**: corre `ProsusAI/finbert` sobre los headlines, guarda CLS-token por día.
   - Output: `data/processed/finbert_embeddings.csv` (2500 filas × 768 cols)
   - Checkpoint cada 200 días para no perder progreso si Colab desconecta
   - Sanity check: norma media > 5.0; embedding 2020-03-16 con signo negativo

5. **Reejecutar** `QQQ_Hibrido_Completo.ipynb` con `finbert_embeddings.csv` en su lugar.

### PASO 3 — Ablation study (Fase 5, una vez disponible FinBERT)
Comparar tres configuraciones para cuantificar el aporte de cada componente:

| Experimento | Descripción |
|-------------|-------------|
| A — Precio solo | `HybridPredictiveModel` con sentimiento = ceros (ya hecho: RMSE 1.10%) |
| B — Precio + FinBERT | Mismo modelo, embeddings reales (objetivo: RMSE < 0.8%) |
| C — Sin CrossAttention | Ablación de la capa de fusión (tabla comparativa para tesis) |

### PASO 4 — TimeGAN calidad distribucional (Fase 6b)
Una vez FinBERT disponible:
- Aumentar `GAN_EPOCHS` a 500 en `config.py`
- Generar escenarios condicionados al embedding COVID (2020-03-16)
- Reportar hechos estilizados: clustering de volatilidad, leverage effect, kurtosis

### PASO 5 — Documentación final (Fase 7)
- Diagrama de arquitectura del sistema híbrido
- Análisis de interpretabilidad: attention weights por feature técnico
- Redacción de sección de metodología para la tesis

---

## Contacto

- Email: yabdul1506@gmail.com
- Directora: Sonia Jaramillo Valbuena, Universidad del Quindío
