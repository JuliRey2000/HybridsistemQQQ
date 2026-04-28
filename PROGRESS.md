# Progreso del Proyecto

## Estado General: PROTOTIPO FUNCIONAL EN CONSTRUCCIÓN 🔄

**Fecha de Inicio:** Abril 2026
**Último Actualizado:** Abril 27, 2026
**Timeline:** 2-3 meses para completar todas las fases

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
- [ ] Análisis de umbral óptimo (sensitivity analysis)
- [ ] Robustez a diferentes regímenes de mercado (alcista 2017-2019, caída 2022)

---

## ⏳ FASE 6b: TimeGAN — Módulo Generativo (ESTRUCTURA LISTA)

Cubierta en `QQQ_Hibrido_Completo.ipynb` Sección 6:

- [x] Arquitectura `TimeGANGenerator + WassersteinCritic` implementada
- [x] Entrenamiento WGAN-GP con n_critic=5, λ_gp=10
- [x] Métricas generativas: Wasserstein Distance, hechos estilizados
- [x] Visualización trayectorias reales vs generadas
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

| Métrica | Baseline | LSTM solo | Híbrido | Target |
|---------|----------|-----------|---------|--------|
| RMSE (%) | 1.5 | 1.0 | 0.8 | < 0.8 |
| MAE (%) | 1.2 | 0.8 | 0.6 | < 0.6 |
| Sharpe | — | 0.3 | 0.5 | > 0.5 |
| Max DD (%) | -25 | -20 | -15 | > -15 |

---

## Problemas Identificados y Soluciones

| Problema | Solución |
|----------|----------|
| Look-ahead bias | ✅ Walk-forward cronológico sin shuffle |
| Data leakage en normalización | ✅ Scaler ajustado solo en train, transform en val/test |
| Sentimiento sin corpus | 🔄 Zeros como placeholder — Fase 4 lo resuelve |
| Mercados volátiles | ✅ Huber Loss en lugar de MSE |
| GAN inestabilidad | ✅ WGAN-GP (Gradient Penalty) en lugar de weight clipping |

---

## Próximo Paso Inmediato

**Construir el corpus FinBERT (Fase 4):**
1. Descargar FNSPID desde Kaggle
2. Ejecutar `compute_embeddings.py` en Colab (GPU T4, ~3h para 2500 días)
3. Subir `finbert_embeddings.csv` a `data/processed/`
4. Reejecutar `QQQ_Hibrido_Completo.ipynb` con sentimiento real

---

## Contacto

- Email: yabdul1506@gmail.com
- Directora: Sonia Jaramillo Valbuena, Universidad del Quindío
