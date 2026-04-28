# Prototipo Funcional Sistema Híbrido QQQ — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Crear el notebook `QQQ_Hibrido_Completo.ipynb` que usa `src/` directamente e implementa el sistema completo: `HybridPredictiveModel` (BiLSTM + CrossAttention + FinBERT) con walk-forward validation, métricas RMSE/MAE/DA/Sharpe para t+1 y t+5, backtesting, y módulo generativo TimeGAN + WGAN-GP.

**Architecture:** El notebook no reimplementa nada — importa desde `src/models`, `src/train`, `src/utils`, `src/data_pipeline`. La validación es walk-forward cronológica (5 folds). El target es regresión (valor de retorno %) en lugar de clasificación binaria.

**Tech Stack:** PyTorch, yfinance, ta, scikit-learn, MLflow/DagsHub, HuggingFace Transformers (opcional para FinBERT real)

---

## Contexto — Por qué este plan

El prototipo de póster (`QQQ_Prototipo_Colab.ipynb`) servía para la presentación: clasifica dirección (sube/baja) con split estático y sin usar `src/`. Ese notebook queda intacto como registro del póster.

El prototipo funcional debe:
- Usar `HybridPredictiveModel` (regresión, t+1 y t+5 simultáneos)
- Usar walk-forward validation desde `utils.walk_forward_splits`
- Cargar datos con `DataPipeline` (incluyendo FinBERT zero-padding si no hay corpus)
- Reportar todas las métricas de tesis: RMSE, MAE, DA, Sharpe, MaxDD
- Incluir módulo generativo TimeGAN + WGAN-GP

## Archivos que cambian

| Acción | Archivo |
|--------|---------|
| CREAR  | `notebooks/QQQ_Hibrido_Completo.ipynb` |
| MODIFICAR | `PROGRESS.md` |
| MODIFICAR | `wiki/resumenes/resumen-proyecto-sistema-hibrido-qqq.md` |
| CREAR  | `wiki/resumenes/resumen-notebook-hibrido-completo.md` |

Los archivos `src/` **no se modifican** — ya están completos y correctos.

---

## Task 1: Notebook QQQ_Hibrido_Completo.ipynb

**Estructura de secciones:**
```
0. Setup & Bootstrap (Colab + MLflow + DagsHub)
1. Pipeline de Datos (DataPipeline con cache en disco)
2. EDA Rápido
3. Walk-Forward Training (HybridPredictiveModel, 5 folds)
4. Evaluación Test Out-of-Sample (vs. Naive y Ridge baselines)
5. Backtesting Long/Short
6. TimeGAN — Módulo Generativo (opcional)
7. Resumen Final
```

**Diferencias clave vs. poster notebook:**

| Aspecto | Poster | Funcional |
|---------|--------|-----------|
| Modelo | `LSTMDirectionModel` (inline) | `HybridPredictiveModel` (desde `src/`) |
| Target | Binario (sube/baja) | Regresión (retorno %, t+1 y t+5) |
| Split | Estático 70/15/15 | Walk-forward 5 folds |
| Métricas | Accuracy, AUC-ROC | RMSE, MAE, DA, Sharpe |
| Generativo | No | TimeGAN + WGAN-GP |
| Sentimiento | No | FinBERT (zeros si sin corpus) |

- [ ] Crear `notebooks/QQQ_Hibrido_Completo.ipynb`

## Task 2: PROGRESS.md

- [ ] Actualizar estado: Fase 1 completa, prototipo funcional iniciado (Fases 2-3-5-6 parcialmente cubiertas en notebook)
- [ ] Agregar nota sobre el cambio de enfoque (de póster a prototipo funcional)
- [ ] Listar próximo paso concreto: corpus FinBERT

## Task 3: Wiki Resúmenes

- [ ] Actualizar `resumen-proyecto-sistema-hibrido-qqq.md`: estado actual (prototipo funcional notebook creado)
- [ ] Crear `resumen-notebook-hibrido-completo.md`: documenta el nuevo notebook

---

## Notas de Implementación

### DataPipeline → 9 features técnicos
`data_pipeline.add_technical_indicators` genera 9 features: RSI_14, MACD, MACD_Signal, MACD_Diff, BB_Pct, ATR_14, SMA_20, SMA_50, Vol_Change. `HybridPredictiveModel` recibe `price_input_size=N_FEATURES` dinámicamente.

### Walk-forward split
`utils.walk_forward_splits(test_start, n_splits=5)` devuelve índices que indexan directamente en los arrays `data['price_seqs']` etc. El test set es `np.arange(test_start, N_SAMPLES)`.

### TimeGAN — ventanas de retornos
El generador trabaja con ventanas univariadas de 20 días de `Daily_Return`. Se construyen desde `price_df.csv` guardado por el pipeline. El sentimiento condiciona al generador (zeros mientras no haya corpus FinBERT).

### MLflow tracking
Run principal cubre el módulo predictivo. TimeGAN puede tener su propio run opcional separado.
