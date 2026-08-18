# Diseño: Búsqueda de Hiperparámetros con Validación Anidada

**Fecha:** 2026-08-18
**Proyecto:** Sistema Híbrido LSTM+FinBERT para predicción de retornos QQQ
**Alcance:** Optimización bayesiana de los hiperparámetros del `HybridPredictiveModel` mediante walk-forward anidado, sin contaminar la validación ni el test out-of-sample

---

## Objetivo

Reemplazar los ~20 hiperparámetros fijos elegidos a mano en `config.py` por una configuración obtenida mediante búsqueda bayesiana con validación anidada, produciendo:

1. `results/best_hparams.json` — la configuración final, reportable y reproducible.
2. Una estimación **insesgada** del procedimiento completo (búsqueda + entrenamiento), que es lo que permite seguir presentando las métricas walk-forward como estimación de generalización.
3. El registro completo de trials en MLflow/DagsHub como evidencia del capítulo de optimización.

**No es objetivo** mejorar el RMSE de forma sustancial. Ver "Expectativa de resultados".

---

## Contexto: por qué anidado y por qué el diseño es conservador

### El problema de contaminación

Hoy `walk_forward_splits(test_start, 5)` corre solo sobre el primer 85%, así que el test está limpio. Pero los 5 folds de validación ya cumplen dos funciones: early stopping y selección del mejor checkpoint. Si además se usan para elegir hiperparámetros, las métricas walk-forward reportadas (RMSE t+1 1.6012%, DA 0.561) dejan de ser estimaciones de generalización y pasan a ser métricas de selección, sesgadas optimistamente.

La validación anidada resuelve esto: la búsqueda ocurre **dentro** del entrenamiento de cada fold externo, y la validación externa nunca participa en la selección.

### El problema de tamaño muestral (determina el resto del diseño)

Partiendo de las 365 muestras de test conocidas, el dataset tiene ~2.433 secuencias:

| Nivel | Tamaño aprox. |
|---|---|
| Test OOS (intocable durante toda la búsqueda) | 365 |
| Train+Val (dominio del walk-forward) | 2.068 |
| Validación de cada fold externo | 138 |
| Validación interna, si fuese split simple | 186–268 |

*(cifras calculadas con la propia `walk_forward_splits` del repositorio, no estimadas)*

**Ancla empírica del propio proyecto**: en el test de 365 muestras, una mejora de RMSE del 0.21% relativo dio Diebold-Mariano p=0.392 — indistinguible de ruido. Con 138–268 muestras el poder estadístico es aún menor.

A esto se suma el *winner's curse*: quedarse con el mejor de N trials equivale a tomar el máximo de N estimaciones ruidosas. Aunque todas las configuraciones fueran equivalentes en calidad real, una ganaría por azar, y su `val_loss` sería optimista por construcción. **Más trials sobre un objetivo ruidoso seleccionan más ruido.**

Tres decisiones del diseño responden directamente a esto: validación interna con 3 folds en vez de split simple, espacio reducido a 6 dimensiones, y regla de un error estándar para desempatar.

---

## Protocolo anidado

### Nivel externo (estimación insesgada)

Para cada uno de los 5 folds externos que ya produce `walk_forward_splits(test_start, n_splits=5)`:

1. Se toma `outer_train_idx` y se corre una **búsqueda interna completa** (ver nivel interno).
2. La configuración ganadora se reentrena sobre `outer_train_idx` **completo**, con el presupuesto de épocas final (100 épocas, patience 15).
3. Se evalúa en `outer_val_idx` y se registran RMSE, DA y `val_loss` para t+1 y t+5.

Las 5 métricas externas resultantes son la estimación insesgada del procedimiento completo. Es esperable y correcto que cada fold externo produzca una configuración distinta; eso se documenta como resultado, no como problema.

### Nivel interno (selección)

Dentro de `outer_train_idx`, se aplica `walk_forward_splits(len(outer_train_idx), n_splits=3, train_min_frac=0.6)` — se reutiliza la función ya existente y probada, sobre el subconjunto de índices.

Cada trial:
1. Entrena y evalúa en los **3 folds internos**, con presupuesto reducido (40 épocas, patience 8).
2. Su puntaje es la **media de `val_loss` de los 3 folds internos**.

Tamaños reales, calculados con la propia `walk_forward_splits` del repositorio:

| Fold externo | `outer_train` | `outer_val` | Validación interna (3 folds) | Split simple habría dado |
|---|---|---|---|---|
| 0 | 1.240 | 138 | 372 (124 × 3) | 186 |
| 1 | 1.378 | 138 | 414 (138 × 3) | 206 |
| 2 | 1.516 | 138 | 453 (151 × 3) | 227 |
| 3 | 1.654 | 138 | 495 (165 × 3) | 248 |
| 4 | 1.792 | 138 | 537 (179 × 3) | 268 |

El esquema de 3 folds internos **duplica** exactamente las muestras de validación por trial (372–537 frente a 186–268), lo que reduce el ruido del objetivo en un factor ≈√2. El beneficio mayor, sin embargo, no es ese: es que la evaluación se promedia sobre **tres períodos de mercado distintos**, lo que penaliza las configuraciones que solo funcionan en un régimen — exactamente el sobreajuste que interesa evitar.

### Búsqueda final y reporte

Terminado el nivel externo, se corre una búsqueda adicional con el mismo protocolo interno sobre **todo el rango 0..test_start**. Su ganadora es la configuración final:

- Se escribe en `results/best_hparams.json`.
- Alimenta el walk-forward de 5 folds existente (flujo actual del notebook).
- Se evalúa **una sola vez** contra el test OOS.

### Escalado sin fuga

En todos los niveles se reutiliza `scale_price_sequences(seqs, fit_idx)` pasando el índice de entrenamiento **del nivel correspondiente**. Es el punto donde es más fácil filtrar información sin notarlo, y está cubierto por un test explícito.

---

## Espacio de búsqueda

Seis dimensiones. Cada dimensión adicional es otra oportunidad de ajustar ruido, así que el espacio se mantiene deliberadamente pequeño.

| Grupo | Hiperparámetro | Distribución |
|---|---|---|
| Optimización | `lr` | log-uniforme(1e-4, 5e-3) |
| | `weight_decay` | log-uniforme(1e-6, 1e-3) |
| Capacidad | `hidden_size` | categórica {64, 128, 256} |
| | `d_model` | categórica {32, 64, 128} |
| Regularización | `dropout` | uniforme(0.1, 0.5) |
| Pérdida | `w_t1` | uniforme(0.3, 0.8), con `w_t5 = 1 − w_t1` |

Cubren los dos capítulos pendientes del documento: optimización (`lr`, `hidden_size`, `d_model`) y mitigación de overfitting (`dropout`, `weight_decay`).

`w_t1`/`w_t5` entran a propósito: hoy son 0.6/0.4 puestos a mano, y el proyecto ya documentó que t+5 rinde mejor que t+1 en RMSE equivalente diario. Dejar que los datos fijen ese peso es material citable.

### Valores fijos y su justificación

| Fijo | Valor | Razón |
|---|---|---|
| `num_heads` | 4 | Efecto esperado menor que el de `d_model`. Además elimina la restricción de divisibilidad: 32, 64 y 128 son todos divisibles por 4, así que ninguna combinación del espacio es inválida. |
| `num_lstm_layers` | 2 | Con ~1.200–1.800 muestras de entrenamiento, más capas aportan poco y aumentan el sobreajuste. |
| `batch_size` | 32 | Interactúa fuertemente con `lr`, que sí se busca; fijarlo evita gastar una dimensión en un efecto redundante. |
| `LOOKBACK` | 30 | Cambiarlo obliga a regenerar las secuencias del dataset — es otro experimento, fuera de alcance. |

---

## Criterio de selección

### Objetivo

`val_loss` (Huber combinado t+1/t+5), promediado sobre los 3 folds internos. Es exactamente lo que el entrenamiento ya minimiza, lo que lo hace la señal más estable, y respeta el diseño de doble cabeza en vez de privilegiar un horizonte.

RMSE, MAE y DA (t+1 y t+5) se registran para **todos** los trials aunque no guíen la optimización, para que el documento pueda mostrar la relación entre el objetivo y las métricas de interés.

### Regla de un error estándar

No se toma ciegamente el mínimo. Procedimiento:

1. Sea `best_mean` la menor media de `val_loss` entre trials completados, y `SE = std(val_loss de sus 3 folds internos) / √3`.
2. Se consideran **candidatas** todas las configuraciones con `mean_val_loss ≤ best_mean + SE`.
3. Entre las candidatas se elige la más parsimoniosa, por este orden de desempate: menor `hidden_size` → menor `d_model` → mayor `dropout` → mayor `weight_decay`.

Es la *one-standard-error rule* (Breiman et al., usada en CART y lasso). Ataca el winner's curse directamente y es defendible en el documento como evidencia de haber tratado el problema de selección, no solo de haber corrido Optuna.

---

## Componentes

### `src/hpo.py` (nuevo)

Módulo autocontenido. Depende de `train.py` y `utils.py`; **nunca al revés**. Se mantiene separado porque `train.py` ya aloja `Trainer` y `GANTrainer` y añadir la búsqueda ahí lo volvería inmanejable.

| Función | Responsabilidad |
|---|---|
| `suggest_hparams(trial)` | Define el espacio de búsqueda; devuelve el dict de hiperparámetros |
| `evaluate_config(hp, data, train_idx, device, n_inner, ...)` | Entrena y evalúa una config sobre los folds internos; devuelve media y desviación de `val_loss` + métricas auxiliares |
| `run_search(data, train_idx, n_trials, study_name, storage, ...)` | Crea o reanuda el estudio Optuna y ejecuta los trials |
| `select_one_se(study)` | Aplica la regla de un error estándar y devuelve la config elegida |
| `nested_walk_forward(data, test_start, ...)` | Orquesta el protocolo de dos niveles completo |
| `estimate_cost(data, train_idx, device)` | Cronometra un trial real y extrapola el costo total |

### `run_hpo.py` (nuevo)

Entrypoint CLI al estilo de `run_train_predictive.py`. Flags: `--trials`, `--inner-folds`, `--mode {nested,final,estimate}`, `--resume`.

### Notebook

Una celda nueva entre las secciones 3 y 4, que carga `best_hparams.json` si existe y en caso contrario usa los valores de `config.py`. El notebook sigue siendo ejecutable sin haber corrido la búsqueda.

### Archivos generados

| Archivo | Contenido |
|---|---|
| `results/hpo.db` | Storage SQLite de Optuna (reanudable) |
| `results/best_hparams.json` | Configuración final elegida |
| `results/hpo_trials.csv` | Todos los trials con sus métricas, para las tablas del documento |
| `results/hpo_nested.csv` | Config ganadora y métricas por fold externo |

---

## Reanudabilidad y manejo de fallos

- **Storage persistente**: `sqlite:///results/hpo.db` con `load_if_exists=True`. Una desconexión de Colab no cuesta horas de cómputo.
- **Pruning**: `MedianPruner` con warmup, que mata los trials cuyo `val_loss` va peor que la mediana. Es lo que hace viable el anidado.
- **Trials fallidos**: un OOM o una configuración inválida devuelve `float("inf")` y se registra; nunca tumba el estudio.
- **Reproducibilidad**: semilla derivada del número de trial, de forma que reanudar produce los mismos resultados.
- **MLflow**: un run padre por búsqueda, con cada trial como run anidado, para no dejar cientos de runs sueltos en DagsHub.

---

## Verificación

Pruebas reales, no comprobación de que compila. Los tests de índices corren localmente (son aritmética pura, sin torch); el smoke run requiere GPU y corre en Colab.

| Test | Qué garantiza |
|---|---|
| **No contaminación** | Ningún índice ≥ `test_start` aparece en ningún split de la búsqueda, en ningún nivel. Es el bug que invalidaría la tesis entera. |
| **No look-ahead** | En ambos niveles, todo índice de validación es cronológicamente posterior a todo índice de su entrenamiento. |
| **Aislamiento del fold externo** | `outer_val_idx` no aparece en ningún split interno de su propio fold. |
| **Validez del espacio** | Toda combinación muestreada satisface `d_model % num_heads == 0` y produce un modelo instanciable. |
| **Regla de 1-SE** | Sobre un estudio sintético con óptimo conocido, la regla elige la config parsimoniosa esperada y no el mínimo crudo. |
| **Reanudabilidad** | Interrumpir el estudio y reanudarlo continúa desde el trial correcto sin repetir. |
| **Smoke run** | 2 trials × 2 épocas de extremo a extremo produce un `best_hparams.json` válido. |

---

## Expectativa de resultados

**La búsqueda no va a mover el RMSE de forma sustancial.** En series financieras diarias la ganancia típica de afinar hiperparámetros es de un pequeño porcentaje relativo, y el propio Diebold-Mariano del proyecto indica que el modelo no le gana al Naive de forma significativa. Lo que mueve esa aguja es el corpus FinBERT, no los hiperparámetros.

El valor de este módulo es **completar el capítulo metodológico**: pasar de "valores elegidos a mano, sin justificación" a "búsqueda bayesiana con validación anidada y regla de un error estándar". Conviene que el documento presente el resultado en esos términos y no prometa ganancias de rendimiento.

---

## Costo estimado

Referencia: **40 trials por búsqueda** (el default), 3 folds internos, presupuesto reducido de épocas y pruning activo → **~15 horas de T4**, repartibles en varias sesiones gracias al storage SQLite.

La estimación **no está medida**. `run_hpo.py --mode estimate` cronometra un trial real y extrapola el costo antes de comprometer horas; el número definitivo de trials se decide con ese dato en la mano.

Si hace falta recortar, la palanca es **bajar los trials** (25 en vez de 40), nunca reducir los folds internos: pocas configuraciones bien evaluadas valen más que muchas mal evaluadas.

---

## Dependencias

```
optuna>=3.4.0        # búsqueda bayesiana (TPE) + pruning + storage
```

Se añade a `requirements.txt`. Optuna no requiere GPU propia ni credenciales.

---

## Orden de ejecución

```bash
# 1. Estimar el costo real de un trial antes de comprometer horas
python run_hpo.py --mode estimate

# 2. Validación anidada — la estimación insesgada del procedimiento
python run_hpo.py --mode nested --trials 40 --inner-folds 3

# 3. Búsqueda final sobre todo el train+val -> best_hparams.json
python run_hpo.py --mode final --trials 40 --inner-folds 3

# 4. Reentrenar el walk-forward con la config ganadora y evaluar en test
#    (notebook, secciones 3-5)
```

Reanudar cualquier búsqueda interrumpida: añadir `--resume`.

---

## Fuera de alcance

- **Ejecución real de la búsqueda**: se lanza cuando existan los embeddings FinBERT reales. Hoy la rama de sentimiento recibe ceros, y los óptimos de `d_model`, `dropout` y `w_t1` cambiarán al activarla. Tunear ahora gastaría GPU dos veces.
- **Modificar `config.py`**: los valores actuales siguen siendo el default reproducible. La configuración ganadora vive en `best_hparams.json` hasta que se decida promoverla.
- **`LOOKBACK` y arquitectura**: cambiar la ventana o la topología del modelo son experimentos distintos.
- **Ablation study** (rama de precio sola vs. híbrida) y **stress-test con escenarios del TimeGAN**: son los otros dos huecos identificados del documento, con su propio diseño.
