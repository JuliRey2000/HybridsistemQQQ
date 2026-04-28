# 🚀 Guía de Inicio Rápido

Sigue estos pasos para comenzar con el proyecto.

---

## 1️⃣ Instalación (5 minutos)

```bash
# Navega a la carpeta del proyecto
cd "Prototipo sistema tesis_master"

# Crea un ambiente virtual
python -m venv venv

# Actívalo
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# Instala dependencias
pip install -r requirements.txt

# Verifica la instalación
python -c "import torch; import pandas; print('✓ OK')"
```

---

## 2️⃣ Configuración (2 minutos)

```bash
# Copia el archivo de configuración de ejemplo
# Windows:
copy .env.example .env
# Mac/Linux:
cp .env.example .env

# Abre .env y ajusta valores si es necesario
# (por defecto están bien para empezar)
```

---

## 3️⃣ Ejecutar Pipeline de Datos (10-15 minutos)

```bash
# Descarga datos, calcula indicadores, crea secuencias LSTM
python run_pipeline.py

# Esto descargará ~5 años de datos QQQ
# Generará ~1000+ secuencias de 30 días cada una
# Guardará en data/processed/
```

**Esperado:**
- Verás logs mostrando progreso
- Al final: "PIPELINE COMPLETADO EXITOSAMENTE ✓"
- Archivos creados en `data/processed/`:
  - `X_sequences.npy` (secuencias LSTM)
  - `y_targets.npy` (retornos objetivo)
  - `df_processed.csv` (datos completos)

---

## 4️⃣ Exploración de Datos (FASE 2)

Una vez completado el pipeline:

```bash
# Abre el notebook educativo
jupyter notebook notebooks/01_eda.ipynb
```

Este notebook te mostrará:
- Distribución de retornos
- Autocorrelación en series temporales
- Correlación entre features
- Por qué necesitamos LSTM

---

## 5️⃣ Entrenar Modelo LSTM (FASE 3)

Después de exploración:

```bash
# Abre el notebook de entrenamiento
jupyter notebook notebooks/03_lstm_training.ipynb
```

Este notebook:
- Carga datos preparados
- Entrena BiLSTM
- Muestra métricas
- Visualiza predicciones

---

## 📊 Estructura Actual

```
proyecto/
├── data/
│   ├── raw/              # ← Datos descargados (se llena al correr run_pipeline.py)
│   └── processed/        # ← Datos procesados (X, y, scaler)
├── src/
│   ├── data_pipeline.py  # ← Lee aquí para entender el flujo
│   ├── models.py         # ← Arquitecturas neuronales
│   ├── train.py          # ← Bucle de entrenamiento
│   └── utils.py          # ← Funciones auxiliares
├── notebooks/            # ← Notebooks educativos (FASE 2, 3, 4)
├── models/               # ← Modelos entrenados (.pth)
├── results/              # ← Gráficos y métricas
├── run_pipeline.py       # ← EJECUTA ESTO PRIMERO
├── config.py             # ← Configuración centralizada
└── README.md             # ← Documentación completa
```

---

## ❓ Preguntas Frecuentes

### ¿Cuánto tarda el pipeline?
- Descarga datos: 1-2 min
- Cálculo indicadores: 2-3 min
- Normalización y secuencias: 1-2 min
- **Total:** 5-10 minutos

### ¿Necesito GPU?
No es obligatorio. LSTM entrenará en CPU (más lento).
Para GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ¿Tengo que ajustar .env?
No, los valores por defecto son buenos para empezar.

### ¿Qué si falla la descarga de datos?
- Verifica conexión a internet
- Prueba directamente:
  ```python
  import yfinance as yf
  data = yf.download('QQQ', start='2024-01-01', end='2024-01-31')
  print(len(data))  # Debería mostrar > 0
  ```

### ¿Dónde veo los logs?
- Consola en tiempo real
- Archivo: `logs/pipeline.log`

---

## 🎯 Próximos Pasos

**Después de completar el pipeline:**

1. ✅ Ejecuta `python run_pipeline.py`
2. ⏭️ Abre `notebooks/01_eda.ipynb` para entender los datos
3. ⏭️ Abre `notebooks/03_lstm_training.ipynb` para entrenar
4. ⏭️ Luego continúa con FASE 4 (Sentimiento) y 5 (Híbrido)

---

## 📞 Si Tienes Problemas

1. **Lee el README.md** - Tiene sección "Troubleshooting"
2. **Revisa los logs** en `logs/pipeline.log`
3. **Verifica instalación** de dependencias
4. **Contacta:** yabdul1506@gmail.com

---

## 🎓 Aprendizaje Recomendado

Mientras ejecutas el pipeline, aprende estos conceptos:

- **Series Temporales:** Qué son, por qué el orden importa
- **LSTM:** Por qué es mejor que redes normales para secuencias
- **Normalización:** Por qué es importante (sin mirar el futuro)
- **Early Stopping:** Cómo evitar overfitting

Excelentes recursos:
- Video LSTM: https://www.youtube.com/watch?v=8HyCNIVRwUI
- Blog LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

---

**¡Listo para comenzar? Ejecuta:**
```bash
python run_pipeline.py
```

**Estimado tiempo:** 5-10 minutos ⏱️
