# Progreso del Proyecto

## Estado General: PROTOTIPO FUNCIONAL VALIDADO ✅ — BLOQUEANTE: CORPUS FINBERT

**Fecha de Inicio:** Abril 2026
**Último Actualizado:** Julio 4, 2026

---

## ▶ ARRANQUE MAÑANA — indicación única: **"ejecuta el PASO 2b"**

Con esa sola frase, Claude implementa en este orden (local → commit → push):

1. **Ensemble de folds en test**: guardar `hybrid_fold{k}.pth` + scaler de cada fold en el
   walk-forward (`cell-wf-train`) y en `cell-test-eval` promediar las predicciones de los
   5 modelos (cada uno con su scaler). Tabla comparativa: mejor fold vs ensemble.
2. **Significancia estadística** en la evaluación de test: Pesaran-Timmermann sobre DA
   (t+1 y t+5) + Diebold-Mariano del híbrido vs Naive y vs Ridge.
3. **Fila Naive/Ridge** citada junto al RMSE del híbrido con mejora % (insumo para la
   discusión del target 0.8%).
4. **Tabla fold → régimen**: fechas de validación de cada fold con su RMSE/DA (2018,
   COVID 2020, bear 2022...).
5. Documentar `GAN_EPOCHS=500` para la corrida generativa final (vía `.env` en Colab).

**Luego tú, en Colab (~40-60 min T4):**
1. Abrir `QQQ_Hibrido_Completo.ipynb` → ejecutar celda 0 (hace `git pull`).
2. Run all hasta la Sección 5 incluida (el GAN no hace falta para validar el ensemble).
3. Pegar aquí la salida del resumen (como hoy).

**Criterio de éxito:** RMSE t+1 ensemble ≤ 1.1073% y DA ≥ 0.586; tests con p < 0.05.

**Pendientes que solo tú puedes destrabar (cuando puedas, no bloquean lo de mañana):**
- **Corpus FinBERT** (el bloqueante real del RMSE): `kaggle.json` + `TIINGO_API_KEY` →
  `python run_corpus.py` (~3h T4).
- **Sanity check MLflow (2 min)**: confirmar en DagsHub que `test_da_t1`/`test_sharpe_t1`
  (0.586 / 1.203) provienen del run nuevo y no de caché.
- **Mensaje a Sonia**: proponer reformular el target RMSE < 0.8% como mejora relativa al
  baseline + targets DA/Sharpe (ver nota en "Última Corrida — 2026-07-04").
**Timeline:** 2-3 meses para completar todas las fases

---

## Cambios 2026-06-12 — Correcciones metodológicas (REQUIERE RE-EJECUCIÓN EN COLAB)

Dos bugs metodológicos reales corregidos + mejoras, y en la sesión 2 del mismo día se
redefinió el target `y_t5` como retorno acumulado de 5 días (ver más abajo). **Las
métricas del 2026-05-07 quedan invalidadas como referencia final**: se calcularon con
features sin normalizar, con los pesos de la última época (no los del early stopping)
y con la definición antigua de `y_t5`. **Re-ejecutado en Colab el 2026-07-04** — las
métricas corregidas están en la sección "Última Corrida — 2026-07-04" y son las citables.

### Bug 1 — Features sin normalizar (data leakage doc vs realidad)
`fit_scalers()` existía en `data_pipeline.py` pero **nunca se llamaba**: el modelo entrenaba
con SMA_20/SMA_50/ATR/MACD en USD (~100→530, crecientes 2015→2024), RSI en [0,100] y
BB_Pct en [0,1] mezclados. El test set (2024) quedaba fuera de la distribución de
entrenamiento. **Fix:** `scale_price_sequences()` / `transform_price_sequences()` en
`src/utils.py` — StandardScaler ajustado por fold SOLO con ventanas de train (sin leakage),
aplicado en notebook (`cell-wf-train`, `cell-test-eval`) y en `run_train_predictive.py`.
El scaler del mejor fold se guarda en `models/hybrid_best_scaler.joblib` y el test se
transforma con él. El baseline Ridge usa su propio scaler (train+val).

### Bug 2 — Métricas por fold con pesos de la última época
`Trainer.fit` guardaba el mejor checkpoint pero no lo restauraba al modelo; las métricas
por fold (y el modelo en memoria) correspondían a la última época (hasta `patience`=15
épocas pasado el óptimo). **Fix:** `Trainer.fit` ahora restaura el best state_dict al final.

### Bug 3 — Alineación sentimiento↔ventanas en TimeGAN
`gan_sents[i]` usaba `sentiments[i]`, pero la ventana GAN i empieza en `price_df[i]` y
`sentiments[j]` corresponde a `price_df[LOOKBACK + j]`. Con ceros no tenía efecto, pero
habría desalineado el conditioning al integrar FinBERT real. **Fix:** offset `i - LOOKBACK`.

### Mejoras
- **EDA estadístico** (cierra pendientes de Fase 2): test ADF de estacionariedad
  (retornos vs precio), ACF/PACF de retornos y |retornos| (volatility clustering),
  matriz de correlación de los 10 features con detección de pares |r| > 0.9.
- **Selección de THRESHOLD** (cierra pendiente de Fase 6): tabla de sensibilidad con
  métricas completas por umbral (Sharpe, Sortino, MaxDD, retorno, win rate, exposición)
  y recomendación automática (máx. Sharpe con exposición ≥ 30%). Se loguea a MLflow.
- Guard en `cell-pipeline`: aborta si la caché `.npy` tiene < 10 features (pre-VIX).
- `statsmodels` añadido a `requirements.txt` y a la celda de instalación.
- Artefactos MLflow nuevos: scaler del mejor fold + figuras EDA.

### PASO 1 (verificación de features) — RESUELTO
Confirmado contando en `src/data_pipeline.py`: `create_sequences` excluye
`[Daily_Return, Close, Open, High, Low, Volume]`, quedando **10 features** =
9 indicadores técnicos (`RSI_14, MACD, MACD_Signal, MACD_Diff, BB_Pct, ATR_14,
SMA_20, SMA_50, Vol_Change`) + `VIX_Close`. Comentarios "9 features" corregidos
en el notebook; `CLAUDE.md` ya decía 10.

### Cambio de target (sesión 2) — `y_t5` ahora es retorno ACUMULADO de 5 días — RESUELTO
`y_t5` era el retorno **puntual** del día t+5 (`Daily_Return.iloc[i+5]`): predecir el
retorno de un único día a 5 días vista tiene señal casi nula (RMSE t+5 ≈ RMSE t+1 lo
confirmaba). Ahora `create_sequences` lo define como la **suma de los 5 log-returns
siguientes**, equivalente a `100·ln(P_{t+5}/P_t)` — la definición estándar de horizonte
multi-día. Decisiones de implementación:

- **Pérdida** (`src/train.py`): el residuo t+5 se divide por √5 (`t5_scale`) dentro del
  Huber. La std del acumulado es ~√5× la diaria; sin esta normalización el término t+5
  dominaría el gradiente y el early stopping, degradando la métrica principal t+1.
  Los pesos `W_T1=0.6 / W_T5=0.4` conservan así su significado, y el delta de Huber
  opera en la misma escala en ambos términos. Las predicciones del modelo siguen en
  escala acumulada natural.
- **Escala del RMSE t+5**: vive ahora en escala ~√5 (≈ 2.2×). Se reporta también el
  **equivalente diario** (RMSE/√5) en notebook, `run_train_predictive.py` y MLflow
  (`test_rmse_t5_daily_eq`) — ese es el número comparable con t+1. El target de la
  tesis (RMSE < 0.8%) siempre se evaluó sobre t+1 y no se ve afectado.
- **Guard de caché** en `cell-pipeline`: aborta si `std(y_t5) ≈ std(y_t1)` (delata un
  `.npy` con la definición antigua). En Colab no aplica (el pipeline corre fresco al
  clonar), pero protege entornos con caché persistida.
- **Backtest t+5** (`cell-backtest`): marcado como solo referencia direccional —
  compone retornos acumulados de 5 días solapados con paso diario, así que su
  Sharpe/MaxDD quedan inflados.
- Los RMSE t+5 históricos (2026-05-07) usan la definición antigua: **no comparables**.
- Verificado con test sintético: `y_t5[i] == 100·ln(P_{i+5}/P_i)` exacto.

Para validar con Sonia: que la tesis defina el horizonte t+5 como acumulado (es la
convención estándar). Si requiriera el puntual, revertir es una línea en
`create_sequences`.

---

## Última Corrida — 2026-07-04 (post-correcciones) ✅ MÉTRICAS CITABLES

**Notebook:** `QQQ_Hibrido_Completo.ipynb` | **Entorno:** Google Colab T4 | **FinBERT:** ceros (sin corpus)
**Incluye:** normalización por fold sin leakage, restauración del mejor checkpoint, `y_t5` = retorno acumulado de 5 días.
**Modelo:** 354,530 parámetros | **Test:** 365 muestras (15% final) | **Período:** 2015-01-01 → 2024-12-31

| Métrica | Walk-Forward (avg ± std) | Test Out-of-Sample |
|---------|--------------------------|-------------------|
| RMSE t+1 | 1.6012% ± 0.4766% | **1.1073%** |
| DA   t+1 | 0.561 | **0.586** |
| RMSE t+5 (acumulado) | 3.3247% ± 1.0717% | **2.4483%** |
| RMSE t+5 (equiv. diario) | 1.4868% | **1.0949%** |
| DA   t+5 | 0.565 | **0.595** |
| Sharpe t+1 | — | **1.203** ✓ |
| MaxDD t+1  | — | **-13.85%** ✓ |

**THRESHOLD recomendado:** 0.00% (máx. Sharpe = 1.203 con exposición ≥ 30%).

**Targets de la tesis:**
- RMSE < 0.8% → ✗ `1.1073%` (brecha 0.31 pp — ver nota sobre realismo del target al final de esta sección)
- Sharpe > 0.5 → ✓ `1.203`
- MaxDD > -20% → ✓ `-13.85%`

**Comparación con la corrida invalidada (2026-05-07):**
- Walk-forward mejoró en dirección: DA t+1 0.547 → **0.561** (efecto de normalización + restauración del mejor checkpoint). DA t+5 = 0.565 con el nuevo target acumulado (el 0.546 histórico medía otra cosa: retorno puntual del día t+5).
- Test t+1: RMSE 1.0969% → 1.1073% (≈ sin cambio, dentro del ruido); DA/Sharpe/MaxDD idénticos (0.586 / 1.203 / -13.85%).
- RMSE t+5 no comparable con el histórico (cambió la definición de `y_t5`). El equivalente diario t+5 (**1.0949%**) es ahora **mejor que t+1**, y su DA (**0.595**) es la más alta del sistema: el horizonte de 5 días es la señal más fuerte — buen argumento central para la tesis.

**Sanity check pendiente (2 min, DagsHub MLflow):** DA/Sharpe/MaxDD de test idénticos a la corrida vieja con RMSE distinto solo es consistente si el patrón de signos de las 365 predicciones no cambió en ningún día (misma dirección → misma equity curve). Es posible (solo cambió la magnitud), pero conviene confirmar que `test_da_t1`/`test_sharpe_t1` provienen del run nuevo y que este loguea `hybrid_best_scaler.joblib` como artefacto.

**Nota sobre el target RMSE < 0.8% (para discutir con Sonia):** el RMSE de un pronóstico ingenuo (predecir siempre retorno 0) equivale a la desviación estándar de los retornos del período — del orden de 1.1–1.2% diaria en el test (la fila "Naive" de la tabla del notebook tiene el número exacto; recuperarla del run). El modelo (1.1073%) ya está pegado a ese techo. Llegar a 0.8% implicaría R²_OOS ≈ 0.5 sobre retornos diarios, muy por encima de lo publicado en la literatura (R²_OOS del 1–5% ya se consideran económicamente significativos). Recomendación: reformular el target como mejora relativa al baseline ingenuo/Ridge (p. ej. ≥ 5–10% de reducción de RMSE) más targets absolutos en DA y Sharpe (ya cumplidos), o evaluar sobre el equivalente diario t+5 (1.0949%).

---

## Corrida 2026-05-07 (actualización 2) — ⚠ INVALIDADA (pre-correcciones; solo referencia histórica)

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
- [x] ACF/PACF de retornos y |retornos| (celda EDA estadística — 2026-06-12)
- [x] Test ADF de estacionariedad formal (retornos vs precio — 2026-06-12)
- [x] Matriz de correlación entre los 10 features (2026-06-12)

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
- [x] Tabla de sensibilidad de THRESHOLD con métricas completas + recomendación automática (2026-06-12; el valor concreto sale de la próxima corrida)
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
| RMSE t+1 (%) | 1.5 | **1.1073** | esperado < 0.9 | < 0.8 |
| DA   t+1 | — | **0.586** | esperado > 0.60 | > 0.55 |
| Sharpe | — | **1.203** ✓ | — | > 0.5 |
| Max DD (%) | -25 | **-13.85** ✓ | — | > -15 |

*Columna "Híbrido (FinBERT)" se completará en Fase 5. Valores actualizados con la corrida citable 2026-07-04; nótese que t+5 supera a t+1: RMSE eq. diario 1.0949% y DA 0.595.*

---

## Problemas Identificados y Soluciones

| Problema | Solución |
|----------|----------|
| Look-ahead bias | ✅ Walk-forward cronológico sin shuffle |
| Data leakage en normalización | ✅ Scaler por fold ajustado solo en train (implementado de verdad el 2026-06-12 — antes `fit_scalers` existía pero nunca se llamaba y los features iban crudos) |
| Métricas de fold con pesos de última época | ✅ `Trainer.fit` restaura el mejor checkpoint (2026-06-12) |
| Sentimiento GAN desalineado con ventanas | ✅ Offset `i - LOOKBACK` en `cell-gan-setup` (2026-06-12) |
| Sentimiento sin corpus | 🔄 Zeros como placeholder — Fase 4 lo resuelve |
| Mercados volátiles | ✅ Huber Loss en lugar de MSE |
| GAN inestabilidad | ✅ WGAN-GP (Gradient Penalty) en lugar de weight clipping |
| Loop infinito en `GANTrainer.train_epoch` | ✅ Batches materializados como lista + índice explícito (2026-05-07) |
| MLflow "run already active" en notebook póster | ✅ Guard `if mlflow.active_run(): mlflow.end_run()` en `cell-23`; `cell-24` duplicada eliminada (2026-05-07) |
| Exposición 0% en análisis de exposición del póster | ✅ `CONFIDENCE_MARGIN` hardcodeado a 0.10 superaba el rango real de probs [0.521–0.528]; ahora adaptativo (percentil 50) (2026-05-07) |
| Clasificador con sesgo alcista puro (0 cortos) | ⚠ Estructural — `LSTMDirectionModel` aprende drift QQQ y produce probs siempre > 0.5. Encuadrar como "timing largo" en la presentación. El regresor híbrido no tiene este problema. |
| `y_t5` era retorno puntual del día t+5 (señal casi nula) | ✅ `y_t5` = retorno acumulado de 5 días; residuo t+5 normalizado por √5 en la pérdida; RMSE t+5 reportado también como equivalente diario (2026-06-12, sesión 2) |

---

## Próximos Pasos — Ordenados por Prioridad

### PASO 1 — ✅ RESUELTO (2026-06-12)
Son **10 features** (9 indicadores técnicos + VIX_Close). Documentación y comentarios corregidos. Ver sección "Cambios 2026-06-12".

### PASO 1b — ✅ RESUELTO (2026-07-04)
Re-ejecutado en Colab T4. Métricas registradas en "Última Corrida — 2026-07-04":
Test RMSE t+1 1.1073%, DA 0.586, Sharpe 1.203, MaxDD −13.85%. Las correcciones mejoraron
la DA de walk-forward (0.547 → 0.561) sin degradar el test.

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

### PASO 2b — Mejoras implementables SIN corpus (plan 2026-07-04, en orden valor/esfuerzo)

Mientras se consiguen las credenciales del corpus, estas mejoras fortalecen el capítulo
de resultados y son implementables ya:

1. **Ensemble de folds en test** (~30 líneas, sin costo extra de entrenamiento): guardar
   checkpoint y scaler de CADA fold del walk-forward (`hybrid_fold{k}.pth`) y en
   `cell-test-eval` promediar las predicciones de los 5 modelos (cada uno transformando
   el test con su propio scaler). Sin leakage: todos entrenan con datos anteriores al test.
   Reduce varianza y típicamente mejora RMSE/DA vs usar solo el mejor fold. Reportar
   ambos (mejor fold vs ensemble) en la misma tabla.
2. **Significancia estadística** (~40 líneas; `statsmodels`/`scipy` ya están):
   - Test de Pesaran-Timmermann sobre DA (¿0.586 y 0.595 > 0.5 estadísticamente?).
   - Test de Diebold-Mariano del híbrido vs Naive y vs Ridge (t+1 y t+5).
   Con n=365 y DA 0.586 debería dar p < 0.01, pero la tesis debe reportarlo formalmente.
3. **Contextualizar el RMSE vs baseline ingenuo**: extraer del run la fila Naive
   (≈ σ de retornos del test) y Ridge de la tabla comparativa del notebook, y reportar
   la mejora % del híbrido. Es el insumo para la discusión del target 0.8% con Sonia.
4. **Tabla de robustez por régimen** (cierra el pendiente de Fase 6): mapear cada fold
   del walk-forward a sus fechas de validación (corrección 2018, COVID 2020, bear 2022…)
   y presentar RMSE/DA por fold junto a su régimen. Explica la brecha WF 1.60% vs test
   1.11%: los folds tempranos entrenan con menos datos y validan en regímenes más duros
   que el test 2023-2024 (alcista, baja volatilidad).
5. **GAN_EPOCHS = 500** (hoy 200 en `config.py`; 1 línea vía `.env` en Colab) — antes de
   la corrida generativa final. Costo: solo tiempo de GPU.
6. *(Opcional)* **Ensemble de semillas** para el número final de la tesis: 3–5 seeds del
   modelo final, reportar media ± std. Números más defendibles en la sustentación.

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
