# Sistema Híbrido de Deep Learning para Predicción de Retornos QQQ

Proyecto de tesis de maestría en Ingeniería con énfasis en Analítica de Datos.

**Autor:** Julian Esteban Castillo Marulanda
**Directora:** Sonia Jaramillo Valbuena
**Universidad:** Universidad del Quindío

---

## 📋 Descripción del Proyecto

Este proyecto construye un **sistema híbrido de Deep Learning** que predice el retorno del ETF QQQ (NASDAQ-100) para el día siguiente, combinando:

- **Series Temporales:** Red neuronal BiLSTM que procesa 30 días de datos históricos (OHLCV + indicadores técnicos)
- **Análisis de Sentimiento:** Procesamiento de noticias financieras usando FinBERT pre-entrenado
- **Modelo Híbrido:** Fusión de ambas ramas para predicción final

---

## 🚀 Quick Start

### 1. Instalación

```bash
# Clonar o descargar el proyecto
cd "Prototipo sistema tesis_master"

# Crear ambiente virtual (recomendado)
python -m venv venv
source venv/Scripts/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Descargar y Procesar Datos (FASE 1)

```bash
python src/data_pipeline.py
```

Esto descargará:
- 5 años de histórico de precios QQQ
- Calculará indicadores técnicos (RSI, MACD, Bollinger Bands, etc.)
- Creará secuencias LSTM para entrenamiento
- Guardará datos procesados en `data/processed/`

### 3. Exploración de Datos (FASE 2)

Abre el notebook educativo:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Este notebook:
- Visualiza distribución de retornos
- Analiza autocorrelación y estacionariedad
- Establece baseline con modelos simples

### 4. Entrenar Modelo LSTM (FASE 3)

```bash
jupyter notebook notebooks/03_lstm_training.ipynb
```

Este notebook:
- Crea DataLoaders
- Entrena BiLSTM con early stopping
- Evalúa desempeño
- Visualiza predicciones vs reales

---

## 📁 Estructura del Proyecto

```
proyecto/
├── data/
│   ├── raw/                    # Datos descargados crudos
│   └── processed/              # Datos preprocesados (secuencias LSTM)
│
├── models/                     # Modelos entrenados (.pth)
│
├── notebooks/
│   ├── 01_eda.ipynb           # Análisis exploratorio
│   ├── 02_baseline.ipynb       # Modelos simples (regresión lineal, árbol)
│   ├── 03_lstm_training.ipynb  # Entrenamiento del LSTM
│   └── 04_hybrid_model.ipynb   # Integración LSTM + Sentimiento
│
├── src/
│   ├── data_pipeline.py       # Descargar y procesar datos
│   ├── models.py              # Arquitecturas (LSTM, Hybrid, Sentiment)
│   ├── train.py               # Bucle de entrenamiento
│   ├── utils.py               # Funciones auxiliares y backtesting
│   └── __init__.py
│
├── results/                    # Métricas, gráficos, predicciones
│
├── logs/                       # Logs de entrenamiento
│
├── requirements.txt            # Dependencias Python
└── README.md                   # Este archivo
```

---

## 🏗️ Fases de Desarrollo

### ✅ FASE 1: Configuración y Data Pipeline
- [x] Estructura del proyecto
- [x] Instalación de dependencias
- [x] Data pipeline (descarga + preprocesamiento)
- **Deliverable:** Dataset listo para entrenamiento

### 📝 FASE 2: Modelos Baseline
- [ ] Análisis exploratorio (EDA)
- [ ] Modelos simples (Naive, Regresión Lineal, Árbol)
- [ ] Motivación para usar LSTM
- **Deliverable:** Baseline RMSE < 1.5%

### 🔄 FASE 3: Modelo LSTM/BiLSTM
- [ ] Implementación BiLSTM
- [ ] Entrenamiento con early stopping
- [ ] Análisis de errores
- **Deliverable:** LSTM RMSE con mejora >20% sobre baseline

### 📰 FASE 4: Análisis de Sentimiento
- [ ] Descargar noticias financieras
- [ ] Extraer sentimiento con FinBERT
- [ ] Integrar features de sentimiento al dataset

### 🔗 FASE 5: Modelo Híbrido
- [ ] Arquitectura multi-rama
- [ ] Entrenamiento end-to-end
- [ ] Comparativa LSTM vs Híbrido
- **Deliverable:** Mejora >10% sobre LSTM solo

### 💰 FASE 6: Backtesting
- [ ] Estrategia simple de trading
- [ ] Cálculo de Sharpe Ratio, Max Drawdown
- [ ] Validación out-of-sample
- **Deliverable:** Sharpe Ratio > 0.5

### 📚 FASE 7: Documentación Final
- [ ] Documentación técnica
- [ ] Interpretabilidad (feature importance)
- [ ] Visualizaciones finales

---

## 🎓 Conceptos Clave (para principiantes)

### ¿Por qué LSTM?

**Problema:** Los datos financieros son **secuencias temporales**. El retorno de mañana no depende solo de hoy, sino de los últimos 30 días.

**Solución:** LSTM (Long Short-Term Memory) es un tipo de red neuronal diseñada especialmente para procesar secuencias, manteniendo memoria de eventos pasados importantes.

**BiLSTM:** Procesa las secuencias en ambas direcciones (pasado → futuro Y futuro → pasado), capturando más patrones.

### ¿Por qué híbrido?

Los precios siguen patrones **técnicos** (OHLCV) pero también son influenciados por **psicología del mercado** (noticias, sentimiento).

Al combinar:
- **LSTM:** Captura patrones técnicos
- **Sentimiento:** Captura psicología del mercado

Obtenemos una predicción más robusta.

---

## 📊 Métricas de Evaluación

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| **RMSE** | Error cuadrático medio | < 1.5% |
| **MAE** | Error absoluto medio | < 1.0% |
| **Sharpe Ratio** | Retorno/Riesgo | > 0.5 |
| **Max Drawdown** | Máxima pérdida | > -20% |
| **Win Rate** | % días ganadores | > 50% |

---

## 💻 Tecnologías Utilizadas

| Tecnología | Uso |
|-----------|-----|
| **PyTorch** | Framework de Deep Learning |
| **Pandas** | Procesamiento de datos |
| **NumPy** | Computación numérica |
| **Scikit-learn** | Modelos ML clásicos, métricas |
| **yfinance** | Descargar datos de Yahoo Finance |
| **Transformers** | FinBERT para análisis de sentimiento |
| **Matplotlib/Seaborn** | Visualización |
| **Jupyter** | Desarrollo interactivo |

---

## 🔧 Instalación en Detalle

### Requisitos Previos
- Python 3.8 o superior
- pip (administrador de paquetes Python)
- Virtual environment (recomendado)

### Paso a Paso

```bash
# 1. Clonar el repositorio
cd "ruta\a\Prototipo sistema tesis_master"

# 2. Crear virtual environment
python -m venv venv

# 3. Activar venv
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Verificar instalación
python -c "import torch; import pandas; print('✓ Instalación correcta')"
```

---

## 📖 Usar el Data Pipeline

```python
from src.data_pipeline import DataPipeline

# Crear pipeline
pipeline = DataPipeline(
    ticker='QQQ',
    start_date='2019-01-01',
    end_date='2024-04-20',
    lookback_days=30
)

# Ejecutar
df_processed, scaler, X, y = pipeline.process()

# Resultados:
# - X: Arrays de secuencias (n_samples, 30 días, 10 features)
# - y: Retornos objetivo (n_samples,)
# - df_processed: DataFrame con todos los datos
# - scaler: StandardScaler para normalizar datos nuevos
```

---

## 🧠 Usar los Modelos

```python
import torch
from src.models import LSTMModel, HybridModel

# Crear modelo LSTM
model = LSTMModel(
    input_size=10,      # 10 features técnicos
    hidden_size=64,     # Tamaño interno
    num_layers=2,       # 2 capas BiLSTM
    dropout=0.2         # Regularización
)

# Forward pass
x = torch.randn(32, 30, 10)  # batch de 32, 30 días, 10 features
output = model(x)             # Predicción
```

---

## 🚂 Entrenar un Modelo

```python
from src.train import Trainer, create_dataloaders

# Crear dataloaders
train_loader, val_loader = create_dataloaders(
    X_train, y_train, X_val, y_val,
    batch_size=32
)

# Crear trainer
trainer = Trainer(
    model=model,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    learning_rate=0.001
)

# Entrenar
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    early_stopping_patience=10
)
```

---

## 📊 Hacer Predicciones

```python
from src.train import predict

# Predicciones en test set
predictions = predict(model, X_test, device='cuda')

# Evaluar
from src.utils import calculate_metrics
metrics = calculate_metrics(y_test, predictions.flatten())
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"MAE: {metrics['MAE']:.4f}")
```

---

## 🔍 Troubleshooting

### Error: "No module named 'torch'"
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: "No data descargado"
```python
# Verificar conexión a internet
import yfinance as yf
test = yf.download('QQQ', start='2024-01-01', end='2024-01-31')
print(test.shape)
```

### Cómo saber si está usando GPU
```python
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## 📚 Referencias Documentación

- **PyTorch:** https://pytorch.org/docs/
- **Transformers/FinBERT:** https://huggingface.co/yiyanghkust/finbert-tone
- **yfinance:** https://github.com/ranaroussi/yfinance
- **Técnicas LSTM:** https://colah.github.io/posts/2015-08-Understanding-LSTMs/

---

## 📝 Notas Importantes

### Separación Temporal
⚠️ **NUNCA shufflear datos al hacer train/val/test split en series temporales.**

Esto causaría "look-ahead bias" - el modelo vería datos futuros durante entrenamiento.

```python
# ❌ INCORRECTO
X_shuffled = np.random.shuffle(X)

# ✅ CORRECTO
X_train, X_val, X_test = X[:70%], X[70%:85%], X[85%:]
```

### Normalización
La normalización se hace con `train_set` ÚNICAMENTE, luego se aplica a val/test.

```python
# ❌ INCORRECTO
scaler.fit(X)  # Fit en todos los datos
X_all = scaler.transform(X)

# ✅ CORRECTO
scaler.fit(X_train)  # Fit solo en train
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)  # Mismo scaler
```

---

## 📞 Soporte

Para preguntas o problemas, contactar a:
- **Email:** yabdul1506@gmail.com
- **Universidad:** Universidad del Quindío
- **Directora de Tesis:** Sonia Jaramillo Valbuena

---

## 📄 Licencia

Este proyecto es parte de una tesis de maestría. Ver `ANEXO 1. EXENCIÓN DE RESPONSABILIDAD` en el documento de tesis.

---

**Última actualización:** Abril 2026
