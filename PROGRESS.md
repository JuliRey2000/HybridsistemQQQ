# Progreso del Proyecto

## Estado General: REPLANTEAMIENTO PROPUESTO — pendiente de aprobacion

**Ultimo Actualizado:** Agosto 20, 2026 (post-mortem y replanteamiento)

### Proximos pasos acordados

1. **Rehacer el corpus** — la VM de Colab murio y se perdieron finbert_embeddings.csv
   y price_df.csv. ~13 min de descarga + ~25 de FinBERT. **Persistir en Drive.**
2. **Fase 0, la prueba de puerta**: regresion de |r_t+1| sobre volatilidad pasada
   mas componentes principales del sentimiento, mirando el R^2 **incremental**.
   Minutos, sin GPU. Va **antes** que el ablation porque su resultado cambia lo
   que se le propone a la direccion.
3. **Ablation** (sentimiento a ceros, mismo periodo y semilla).
4. **Arreglar** el Sharpe de t+5, el hueco de 126 muestras del walk-forward y la
   seleccion del mejor fold.
5. **HPO**, que sigue sin ejecutarse.

### El replanteamiento (2026-08-20)

De predecir el retorno puntual a **caracterizar su distribucion condicional**:
cabezas de cuantiles con perdida pinball en vez de una salida escalar con error
cuadratico. Con pinball el minimo es el cuantil condicional, no la media, asi que
el mecanismo del colapso desaparece. El baseline pasa de "predecir cero" a
GARCH(1,1). El modulo generativo, hoy huerfano, pasa a estimar el mismo objeto
que la rama predictiva. Documento completo publicado; ver log.md del vault.

### Analisis de potencia — la leccion metodologica del trabajo

Con n=327 el umbral de significancia es |r| > 0.108, y la banda de efectos
plausibles segun la literatura (R^2 OOS de 0.5% a 2%) va de 0.071 a 0.141: el
umbral cae **dentro** de la banda. La potencia para detectar r=0.10 es de ~56%;
para el 80% harian falta ~617 observaciones. Debio calcularse ANTES de ejecutar.

**Como enunciar la conclusion**: no "el efecto es cero", sino "el efecto no es lo
bastante grande para ser detectable ni explotable con este diseno".

---

## Estado General: CORPUS RESUELTO ✅ — RESULTADO NULO DIAGNOSTICADO

**Fecha de Inicio:** Abril 2026
**Último Actualizado:** Agosto 20, 2026 (corrida con FinBERT real completada)

---

## ▶ RESULTADO DE LA CORRIDA CON FINBERT REAL (2026-08-20)

**La corrida se completó y el sentimiento estuvo realmente activo**: 2.214 días
alineados de 2.214, 0 muestras en ceros, 0 días forward-filled. La rama de
FinBERT recibió datos reales por primera vez en el proyecto.

**No hay mejora demostrable.**

| | Ensemble 5f | Naive (cero) | σ del test |
|---|---|---|---|
| RMSE t+1 | 1.3946% | 1.3973% | 1.3934% |
| RMSE t+5 (eq. diario) | 1.2946% | 1.3085% | — |
| DM vs Naive t+1 | p = 0.377 | | |
| DM vs Naive t+5 | p = 0.336 | | |

El Diebold-Mariano contra Naive **empeora**: de 0.257 en la corrida anterior a
0.377. Se aleja de la significancia, no se acerca.

### El diagnóstico: colapso a la media incondicional

Cuatro observaciones que parecen distintas y son la misma:

1. **RMSE = σ del período** en las seis ventanas (razón 0.998–1.008)
2. **327 de 327 predicciones positivas** — el signo nunca cambia
3. **DA = tasa base exacta**: 0.532 observado, 0.532 esperado bajo H0
4. **Pesaran-Timmermann en `nan`** — su varianza lleva `p̂(1−p̂)` y `p̂ = 1`

Cuantificado: `std(pred)/std(real)` = **0.0250** en t+1 y **0.0155** en t+5. Las
predicciones se mueven en un rango de 0.14 puntos porcentuales, centradas en
+0.166 (la deriva del activo).

Contabilidad cerrada al cuarto decimal — un modelo que solo emita la constante
`c = 0.1661`:

```
MSE = var(y) + (media(y) − c)² = 1.94157 + (0.1047 − 0.1661)² = 1.94534
RMSE = 1.3948        el ensemble reportó 1.3946
```

**Toda** la ventaja sobre el naive procede del intercepto. Los 354.530
parámetros y los 3,46M de titulares no aportan nada medible. Y no queda señal
residual: Pearson −0.019 (p=0.735), Spearman −0.048 (p=0.383).

### El Sharpe de 1.103 es beta

`Buy & Hold QQQ = 1.103`. Idéntico, con 327 trades y 100% de exposición: con
todas las predicciones del mismo signo, `sign(pred)` deja la estrategia
permanentemente larga. El MaxDD de −17.20% es el del propio QQQ. Y al filtrar
por convicción el rendimiento **empeora** de forma monótona (Sharpe 1.103 →
0.128, win rate 0.532 → 0.449): los días de mayor convicción del modelo son sus
peores días.

### ⚠ No reportar los targets como cumplidos

Sharpe (1.103 > 0.5) y MaxDD (−17.20% > −20%) salen ✓, y con la meta de RMSE de
`CLAUDE.md` (1.5%) pasaría también el RMSE. **Los tres pasarían por motivos
espurios** y es de lo primero que un tribunal con criterio va a mirar.

Además hay **dos objetivos de RMSE distintos** documentados en el proyecto y
ninguno en `config.py`:

- el notebook usa `< 0.8%` → exige R²_OOS ≈ 0.67 con σ = 1.39%. Inalcanzable.
- `CLAUDE.md` dice `< 1.5%` → lo cumple predecir cero (1.3973%). Gratuito.

Como RMSE ≈ σ, un umbral **absoluto** mide la volatilidad del período de
evaluación, no el acierto. El objetivo debe reformularse relativo al baseline
ingenuo y acompañado del contraste DM.

### Qué queda, por orden

1. **Ablation** (sentimiento a ceros, mismo período y semilla). Casi seguro que
   dará lo mismo, pero hay que demostrarlo, no deducirlo. Cierra el hueco #2.
2. **HPO**, que sigue sin ejecutarse. No arregla un colapso a la media, pero
   cierra el hueco metodológico #1.
3. **Reescribir resultados** alrededor del hallazgo real.
4. **Arreglar el Sharpe de t+5**: `long_short_strategy` compone retornos de 5
   días solapados con paso diario y `sharpe_ratio` anualiza con √252 sin
   condiciones. 2.626 × √(1/5) ≈ 1.17, prácticamente el de t+1. Ese 2.63 no
   puede aparecer en ningún sitio.
5. **Tres conversaciones con Sonia**, no dos: el período, las métricas que se
   rehacen, y ahora lo que la tesis afirma.

**Esto no es un trabajo fallido.** Es una hipótesis bien puesta a prueba con un
corpus construido y verificado, un protocolo sin fuga, contraste estadístico
serio y un diagnóstico preciso del porqué. Escrito con precisión, es un capítulo
de resultados sólido — y bastante más honesto que un Sharpe de 2.63 que no
sobrevive a la primera pregunta.

---

## ▶ CORPUS DESCARGADO (2026-08-20) — cobertura OK y un hallazgo sobre FNSPID

**Cobertura confirmada**: `2015-01-01 → 2023-12-31`, 3.460.916 titulares
(15.549.299 filas crudas, 2.510.531 duplicados eliminados). El guard de
cobertura pasó. La ventana de test cae entera dentro de FNSPID.

**El corpus NO es homogéneo en el tiempo.** 2015 trae ~3.123 titulares/día y
2021 ~335: una diferencia de 9,3×, y 2015 por sí solo es un tercio del corpus.
Que 2020 tenga la sexta parte de 2015 descarta que refleje la actividad
informativa real — es cómo se construyó FNSPID. Tabla completa por año en el
wiki ([[fnspid]]).

**Por qué importa**: el embedding diario es un promedio, y el error estándar de
un promedio va con 1/√n. La señal de sentimiento de los primeros años sale ~3×
más suavizada que la de los últimos, y **la ventana de test (finales de 2022 →
2023) está en la zona de menor volumen**. Sin corregirlo, el modelo entrena
sobre sentimiento suave y se evalúa sobre sentimiento ruidoso; si la rama de
FinBERT no aportara, no se podría distinguir si es que el sentimiento no predice
o que la variable cambió de naturaleza entre train y test.

**Decisión: `MAX_NEWS_PER_DAY=300`**, el nivel del año más pobre. Deja 2021
intacto y baja el resto a su altura. No se pierde información que el modelo
estuviera usando (solo recibe el vector medio, nunca el conteo), el submuestreo
es reproducible (semilla 42) y de paso reduce el cómputo de FinBERT ~4×.
Redacción para la tesis: *submuestreo aleatorio con semilla fija a 300
titulares/día para homogeneizar la precisión del estimador diario de sentimiento
a lo largo del período*.

**Guard nuevo — alineación del sentimiento.** `create_sequences` emparejaba el
embedding por fecha exacta y, si no lo encontraba, metía ceros **en silencio**.
Un desajuste de fechas dejaba la rama de sentimiento vacía sin un solo error, y
el chequeo del notebook tampoco lo veía (solo mira si existe *algún* valor
distinto de cero). Ahora exige 95% de cobertura y levanta si no se alcanza.
5 tests nuevos; 56 en verde.

**Notas menores**: hay titulares en los 3.287 días del calendario, no solo
hábiles; las noticias de fin de semana se descartan porque `compute_embeddings`
solo recorre días de mercado y el fallback de ±1 día nunca se activa. Es
defendible pero hay que declararlo en la metodología.

**El notebook incorpora el arranque en dos tramos** como celda fija (la 9):
FNSPID y `build_corpus` por separado, antes de la pasada larga de FinBERT, para
que el volumen del corpus se conozca **antes** de comprometer horas de GPU.

---

## ▶ GUARDAS DEL CORPUS (2026-08-20) — el runbook de la corrida limpia

Antes de relanzar la corrida se verificó el runbook, y dos agujeros la habrían
arruinado **sin dar ningún error**:

1. **El checkpoint de embeddings sobrevivía al cambio de corpus.** La lista de
   borrado no incluía `data/processed/emb_checkpoints/`. `compute_embeddings.py`
   guarda progreso cada 200 días y lo reanudaba sin comprobar nada: los días ya
   calculados con el FNSPID viejo (el que se corta en 2020) se habrían
   conservado, y los nuevos se habrían calculado con el archivo bueno. Dos
   corpus mezclados dentro de la misma serie — justo la heterogeneidad que
   motivó acotar el estudio a 2015-2023.
2. **Un `fnspid_news.csv` corto se daba por bueno para siempre.** Si el archivo
   viejo sigue en disco, `run_corpus.py` lo salta con `[SKIP] output ya existe`
   y `download_fnspid.py` ni arranca, así que su guard de cobertura nunca corre.
   Es el camino exacto que siguió la corrida del 18 de agosto.

**Lo que se cerró:**

- `check_coverage()` levanta `CoverageError` en vez de avisar, y se comprueba
  **antes** de renombrar el temporal: un archivo corto ya no queda en su sitio.
  `FNSPID_ALLOW_SHORT=1` permite seguir a sabiendas.
- Al reutilizar un CSV ya descargado se revalida su última fecha. Para que esa
  revalidación llegue a correr, el paso de FNSPID va marcado `always_run`.
- El checkpoint guarda la huella del corpus (tamaño + mtime) que lo generó y se
  descarta si el corpus cambió, o si no trae huella.
- 15 tests nuevos en `tests/test_corpus_guards.py` (51 en verde en total).

**Runbook de la corrida limpia (Colab):**

1. Celda 0 (`git pull` + limpia `__pycache__` + credenciales de DagsHub).
2. Celda 1 — la nueva celda de limpieza: poner `LIMPIAR_ARTEFACTOS = True` y
   ejecutarla. Borra `fnspid_news.csv*`, `corpus_merged.csv`, `_buckets/`,
   `*.npy`, `finbert_embeddings.csv`, `price_df.csv` y `emb_checkpoints/`.
   Volver a ponerla en `False` después: en `True` cada re-ejecución tira horas
   de FinBERT a la basura.
3. Resto del notebook en orden. La descarga de FNSPID son ~13 min (23.2 GB) y
   FinBERT varias horas, reanudable por checkpoints.

**Qué mirar en la salida, en este orden:**

| # | Qué | Dónde | Criterio |
|---|-----|-------|----------|
| 1 | Cobertura de FNSPID | `REPORTE FNSPID`, "Rango de fechas" | debe llegar a `2023-12-…`; si no, el pipeline ahora se detiene solo |
| 2 | Volumen del corpus | `REPORTE CORPUS MERGED` | total de noticias y media/día — decide si hace falta `MAX_NEWS_PER_DAY` |
| 3 | RMSE con sentimiento real | resumen final | frente a **1.0928%** con ceros: es la comparación que justifica el módulo de sentimiento |
| 4 | Diebold-Mariano vs Naive | bloque de significancia | ¿baja de **0.257** a algo significativo? |

---

## ▶ CAMBIO DE ALCANCE (2026-08-18): el estudio pasa a 2015-2023

**Decisión tomada tras dos hallazgos de la primera ejecución real del corpus.**

**Hallazgo 1 — el archivo de FNSPID elegido se cortaba en 2020.** La primera
corrida descargó `All_external.csv` (5.7 GB) y reportó cobertura
`2015-01-01 → 2020-06-11`: faltaban 3.5 años. Al elegirlo se verificó su esquema
pero no su rango de fechas. Corregido: ahora se usa `nasdaq_exteral_data.csv`
(23.2 GB, cobertura 2003-2023 verificada muestreando siete puntos), leyendo
**únicamente el titular**, con lo que la decisión metodológica original
(titulares, no artículos completos) se mantiene. Se añadió un guard que avisa si
los datos no alcanzan el final del período.

**Hallazgo 2 — Tiingo requiere plan de pago.** Los 12 meses de 2024 devolvieron
`403 Forbidden`: el token es válido, pero la API de noticias no está en el plan
gratuito. Ante la alternativa de pagar (~$10/mes) o acotar el estudio, se acotó.

**Por qué acotar es la mejor opción, no solo la barata:** deja el corpus
**homogéneo**, una sola fuente de titulares para todo el período. Elimina el
riesgo de mezclar titulares de FNSPID (train) con título+descripción de Tiingo
(2024, ventana de test), que habría hecho que el embedding de FinBERT cambiara de
naturaleza **dentro de la ventana de evaluación**.

**Coste:** ~250 muestras de 2433. La ventana de test queda dentro de 2022-2023,
cubierta al 100% por FNSPID.

**Pendiente con Sonia:** el documento dice 2015-2024. El cambio es defendible
("el corpus cubre hasta 2023 y una sola fuente evita heterogeneidad en la ventana
de evaluación") pero debe conocerlo.

**Importante:** todas las métricas citables actuales quedarán obsoletas tras la
próxima corrida — cambian los datos (menos período) y el sentimiento (real en vez
de ceros). La tabla de resultados de la tesis se rehace entera.

---

## ▶ Hallazgos de la corrida 2026-08-18 (previa a los arreglos)

**El "mejor fold" no es reproducible.** Mismo commit y misma semilla que el
12-ago, y sin embargo:

| Métrica | 12-ago | 18-ago |
|---|---|---|
| Mejor fold — Sharpe | **1.203** | **−0.392** |
| Mejor fold — DA t+1 | 0.586 | 0.501 |
| Ensemble — Sharpe | 1.203 | 1.203 |
| Ensemble — DA t+1 | 0.586 | 0.586 |

El ensemble es estable; el mejor fold oscila entre Sharpe +1.203 y −0.392. Como
las "métricas citables" oficiales son de mejor fold, **conviene que la tesis cite
el ensemble**.

**El PT en NaN queda explicado.** Los folds individuales sí dan p-valor
(t+1 p=0.663, t+5 p=3.6e-03); solo el ensemble da NaN. La degeneración la **crea
el promediado**: al promediar 5 modelos las predicciones se encogen hacia la
media y salen casi todas del mismo signo. No es un sesgo alcista del modelo, y se
explica en una frase en la metodología.

**Bug corregido en el resumen.** El bloque "Targets de la tesis (evaluados sobre
el ensemble)" evaluaba Sharpe y MaxDD sobre el **mejor fold**. Reportaba
`Sharpe → ✗ (−0.392)` cuando el ensemble daba **1.203, que sí cumple**.

---

## ▶ BÚSQUEDA DE HIPERPARÁMETROS IMPLEMENTADA (2026-08-18) — lista, pendiente de ejecutar

Cierra el hueco más grande del documento: hasta hoy los ~20 hiperparámetros de
`config.py` eran valores fijos elegidos a mano, sin ninguna búsqueda detrás.

**Diseño**: walk-forward **anidado**. Por cada fold externo se corre una búsqueda
bayesiana completa dentro de su propio train (con walk-forward interno de 3 folds), y
la validación externa nunca participa en la selección. El test OOS queda intocable
durante toda la búsqueda. Una búsqueda final sobre todo el train+val produce la config
reportable en `results/best_hparams.json`.

Spec: `docs/superpowers/specs/2026-08-18-busqueda-hiperparametros-design.md`
Plan: `docs/superpowers/plans/2026-08-18-busqueda-hiperparametros.md`

**Tres decisiones conservadoras**, motivadas por el tamaño muestral (validación externa
de 138 muestras, interna de 372-537, cifras verificadas con la propia `walk_forward_splits`):
1. Validación interna de **3 folds** en vez de split simple — duplica las muestras por
   trial y promedia sobre tres regímenes de mercado.
2. Espacio de **6 dimensiones**, no 9 (`lr`, `weight_decay`, `hidden_size`, `d_model`,
   `dropout`, `w_t1`). Fijos: `num_heads=4`, `num_lstm_layers=2`, `batch_size=32`.
3. **Regla de un error estándar** (Breiman): entre las configs dentro de 1 SE del mejor
   se elige la más parsimoniosa, en vez del mínimo crudo. Ataca el winner's curse.

**Expectativa realista**: esto NO va a mover el RMSE de forma sustancial. Con
DM p=0.392 contra Naive, lo que mueve esa aguja es el corpus FinBERT. El valor de este
módulo es completar el capítulo metodológico; conviene que el documento lo presente así
y no prometa ganancias de rendimiento.

**Verificación**: 36 tests locales en verde, sin GPU. Incluyen no-contaminación del test
(sobre las llamadas reales, no solo la construcción de splits), ausencia de look-ahead en
ambos niveles, la regla de 1-SE, reanudabilidad del estudio tras una interrupción, y el
contrato entre `best_hparams.json` y la celda del notebook.

**Tú, en Colab (cuando exista el corpus FinBERT):**
```bash
pip install optuna==3.6.1
python run_hpo.py --mode estimate            # mide 1 trial real y extrapola el costo
python run_hpo.py --mode nested  --trials 40 # estimación insesgada (~15h estimadas)
python run_hpo.py --mode final   --trials 40 # -> results/best_hparams.json
```
La búsqueda es reanudable: si Colab se desconecta, relanzar el mismo comando continúa
desde donde iba (`results/hpo.db`).

**Por qué esperar al corpus**: hoy la rama de sentimiento recibe ceros, así que los
óptimos de `d_model`, `dropout` y `w_t1` cambiarán al activar FinBERT. Tunear ahora
gastaría GPU dos veces. El notebook funciona igual sin `best_hparams.json`.

---

## ▶ PASO 2b IMPLEMENTADO (2026-07-09) — pendiente: validar en Colab

Implementado localmente, verificado (compilación + tests sintéticos de PT/DM) y pusheado.
Qué cambió:

1. **Ensemble de folds en test**: `cell-wf-train` guarda `hybrid_fold{k}.pth` + scaler de
   cada fold; `cell-test-eval` promedia las predicciones de los 5 modelos (cada uno
   transformando el test con su propio scaler) y la tabla comparativa reporta mejor fold
   vs ensemble (t+1 y t+5). Métricas MLflow nuevas con sufijo `_ens`.
2. **Significancia estadística** (`cell-stat-tests`, celda nueva tras la evaluación):
   Pesaran-Timmermann sobre la DA + Diebold-Mariano (corrección HLN; para t+5 la varianza
   de largo plazo usa h−1=4 rezagos por el solapamiento) del híbrido vs Naive y vs Ridge,
   para mejor fold y ensemble. Funciones nuevas `pesaran_timmermann()` y
   `diebold_mariano()` en `src/utils.py`, validadas contra cálculo manual independiente.
3. **Fila Naive/Ridge con mejora %**: `cell-test-eval` imprime σ del test (= RMSE Naive)
   y la mejora % de RMSE del híbrido vs Naive y vs Ridge (t+1/t+5, mejor fold y ensemble);
   logueado a MLflow (`improve_pct_vs_naive_t1_ens`, etc.). Tabla completa en
   `comparativa_test.csv` (artefacto).
4. **Tabla fold → régimen** (`cell-fold-regime`, celda nueva tras el resumen de folds):
   fechas de validación por fold con RMSE/DA, σ diaria y retorno del período + etiqueta
   de régimen; contexto del test para el contraste. `robustez_por_regimen.csv` como
   artefacto MLflow. Cierra el pendiente de robustez de Fase 6.
5. **GAN_EPOCHS=500 documentado**: `cell-gan-train` ahora lee `GAN_EPOCHS` del entorno
   (`%env GAN_EPOCHS=500` antes de la celda); notas en la sección 6 del notebook,
   `.env.example` y `config.py`.

El resumen final (`cell-summary`) ahora imprime mejor fold vs ensemble, p-values PT/DM y
mejora % vs baselines — esa es la salida a pegar aquí.

**Tú, en Colab (~40-60 min T4):**
1. Abrir `QQQ_Hibrido_Completo.ipynb` → ejecutar celda 0 (hace `git pull`).
2. Run all hasta la Sección 5 incluida (el GAN no hace falta para validar el ensemble).
3. Pegar aquí la salida del resumen (como la vez pasada).

**Criterio de éxito:** RMSE t+1 ensemble ≤ 1.1073% y DA ≥ 0.586; tests con p < 0.05.

**Pendientes que solo tú puedes destrabar (cuando puedas, no bloquean la corrida):**
- **Corpus FinBERT** (el bloqueante real del RMSE): `kaggle.json` + `TIINGO_API_KEY` →
  `python run_corpus.py` (~3h T4).
- **Sanity check MLflow (2 min)**: confirmar en DagsHub que `test_da_t1`/`test_sharpe_t1`
  (0.586 / 1.203) provienen del run nuevo y no de caché.
- **Mensaje a Sonia**: proponer reformular el target RMSE < 0.8% como mejora relativa al
  baseline + targets DA/Sharpe (ver nota en "Última Corrida — 2026-07-04").
**Timeline:** 2-3 meses para completar todas las fases

---

## ▶ Corrida 2026-08-12 — validación de PASO 2b ejecutada — ⚠ NO CITABLE AÚN, en análisis

Primera corrida en Colab del commit `8e0327c` (PASO 2b). Resultados y análisis, pendiente de
confirmación antes de reemplazar la sección "Última Corrida — 2026-07-04" como referencia oficial.

**Números:**
- WF: RMSE t+1 1.6026% ± 0.4781% (≈ igual a julio), DA t+1 **0.547** (bajó de 0.561 en julio
  pese a seed fijo — ver nota de reproducibilidad abajo).
- Test OOS ensemble t+1: RMSE **1.0945%**, DA 0.586 (mejora sobre mejor-fold 1.1073%/0.586 —
  cumple el criterio numérico pre-registrado).
- Test OOS ensemble t+5: RMSE eq. diario 1.0880%, DA 0.595.
- Sharpe/MaxDD ensemble = mejor fold (1.203 / -13.85%) — mismo patrón direccional de señales.

**Hallazgo 1 — mejora vs Naive NO es estadísticamente significativa:** Diebold-Mariano ensemble
t+1 vs Naive p=0.392 (n.s.); vs Ridge p=2.78e-04 (sí). La mejora de RMSE vs Naive es de solo
+0.21% relativo. Confirma con rigor estadístico que, sin FinBERT, el modelo no le gana al azar
de forma sólida — refuerza la prioridad del corpus.

**Hallazgo 2 — Pesaran-Timmermann da NaN en las 4 combinaciones (mejor fold/ensemble × t1/t5):**
Causa diagnosticada en `src/utils.py:256-261` — el test degenera cuando las predicciones son
casi todas del mismo signo. Sugiere sesgo direccional del modelo (coherente con el sesgo alcista
ya documentado en el clasificador del póster). **Pendiente**: confirmar con el diagnóstico
"Pred > 0 / Pred < 0" que ya imprime `cell-backtest` (celda 21) en el output completo de Colab.

**Hallazgo 3 — nota de reproducibilidad:** `cell-imports` fija SEED=42 + cudnn determinista;
por eso el mejor-fold de esta corrida reproduce exacto julio (no es caché ni bug — verificado).
El leve movimiento del WF DA con RMSE casi idéntico es consistente con no-determinismo conocido
de kernels LSTM en GPU no cubiertos al 100% por `cudnn.deterministic` — DA (métrica de signo) es
más sensible a esto que RMSE. Documentar como limitación de reproducibilidad en la metodología.

**Pendiente antes de citar esta corrida:** confirmar Hallazgo 2, decidir si corregir
`pesaran_timmermann()` o documentarlo como limitación, y decidir si esta corrida reemplaza a la
de 2026-07-04 como referencia oficial.

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
- [x] Robustez a diferentes regímenes de mercado: tabla fold → régimen en `cell-fold-regime` (2026-07-09; los números salen de la próxima corrida en Colab)

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

### PASO 2b — ✅ IMPLEMENTADO (2026-07-09; pendiente validar con la corrida en Colab)

Los puntos 1-5 quedaron implementados — detalle en la sección "PASO 2b IMPLEMENTADO"
al inicio de este documento:

1. ✅ **Ensemble de folds en test** — checkpoint + scaler por fold; promedio de los 5
   modelos en test; tabla mejor fold vs ensemble.
2. ✅ **Significancia estadística** — Pesaran-Timmermann y Diebold-Mariano (HLN) en
   `src/utils.py` + `cell-stat-tests`; verificados con tests sintéticos locales.
3. ✅ **RMSE vs baselines con mejora %** — σ del test = RMSE Naive explícito; mejora %
   vs Naive/Ridge impresa y en MLflow.
4. ✅ **Tabla de robustez por régimen** — `cell-fold-regime` + `robustez_por_regimen.csv`.
5. ✅ **GAN_EPOCHS = 500 documentado** — vía entorno en `cell-gan-train`, `.env.example`
   y `config.py`.
6. *(Opcional, sigue pendiente)* **Ensemble de semillas** para el número final de la
   tesis: 3–5 seeds del modelo final, reportar media ± std. Números más defendibles en
   la sustentación.

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
