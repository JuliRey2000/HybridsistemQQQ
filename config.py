"""
Configuración centralizada del proyecto.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT       = Path(__file__).parent
DATA_RAW_PATH      = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH= PROJECT_ROOT / "data" / "processed"
MODELS_PATH        = PROJECT_ROOT / "models"
RESULTS_PATH       = PROJECT_ROOT / "results"
LOGS_PATH          = PROJECT_ROOT / "logs"

for p in [DATA_RAW_PATH, DATA_PROCESSED_PATH, MODELS_PATH, RESULTS_PATH, LOGS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

# ─── DATA PIPELINE ───────────────────────────────────────────────────────────
TICKER      = os.getenv("TICKER", "QQQ")
START_DATE  = os.getenv("START_DATE", "2015-01-01")   # tesis: 2015-2024
END_DATE    = os.getenv("END_DATE",   "2024-12-31")
LOOKBACK    = int(os.getenv("LOOKBACK", "30"))         # ventana deslizante
HORIZON_T1  = 1                                        # predicción t+1
HORIZON_T5  = 5                                        # predicción t+5

# ─── SENTIMIENTO ─────────────────────────────────────────────────────────────
SENTIMENT_DIM       = 768                              # embedding FinBERT (CLS)
FINBERT_MODEL       = "yiyanghkust/finbert-tone"
SENTIMENT_PRECOMP   = str(DATA_PROCESSED_PATH / "finbert_embeddings.csv")
NEWS_CSV_PATH       = str(DATA_RAW_PATH / "news.csv")

# ─── MÓDULO PREDICTIVO ───────────────────────────────────────────────────────
PRICE_INPUT_SIZE = int(os.getenv("PRICE_INPUT_SIZE", "10"))  # 9 técnicos + VIX_Close
LSTM_HIDDEN_SIZE = int(os.getenv("LSTM_HIDDEN_SIZE", "128"))
LSTM_NUM_LAYERS  = int(os.getenv("LSTM_NUM_LAYERS",  "2"))
D_MODEL          = int(os.getenv("D_MODEL",          "64"))
NUM_HEADS        = int(os.getenv("NUM_HEADS",        "4"))
DROPOUT          = float(os.getenv("DROPOUT",        "0.2"))
W_T1             = float(os.getenv("W_T1",           "0.6"))  # peso pérdida t+1
W_T5             = float(os.getenv("W_T5",           "0.4"))  # peso pérdida t+5

# ─── ENTRENAMIENTO PREDICTIVO ────────────────────────────────────────────────
BATCH_SIZE    = int(os.getenv("BATCH_SIZE",    "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE","0.001"))
WEIGHT_DECAY  = float(os.getenv("WEIGHT_DECAY","0.00001"))
NUM_EPOCHS    = int(os.getenv("NUM_EPOCHS",    "50"))
PATIENCE      = int(os.getenv("PATIENCE",      "10"))

# Walk-forward
WF_SPLITS       = int(os.getenv("WF_SPLITS",       "5"))
WF_TRAIN_MIN    = float(os.getenv("WF_TRAIN_MIN",  "0.6"))
TEST_FRAC       = float(os.getenv("TEST_FRAC",     "0.15"))

# ─── MÓDULO GENERATIVO (TimeGAN + WGAN-GP) ───────────────────────────────────
NOISE_DIM       = int(os.getenv("NOISE_DIM",      "32"))
GAN_HIDDEN      = int(os.getenv("GAN_HIDDEN",     "128"))
GAN_SEQ_LEN     = int(os.getenv("GAN_SEQ_LEN",   "20"))    # días de escenario
GAN_EPOCHS      = int(os.getenv("GAN_EPOCHS",     "200"))
N_CRITIC        = int(os.getenv("N_CRITIC",       "5"))
LAMBDA_GP       = float(os.getenv("LAMBDA_GP",    "10.0"))
LR_GEN          = float(os.getenv("LR_GEN",       "0.0001"))
LR_CRITIC       = float(os.getenv("LR_CRITIC",    "0.0001"))
N_SCENARIOS     = int(os.getenv("N_SCENARIOS",    "100"))

# ─── SISTEMA ─────────────────────────────────────────────────────────────────
DEVICE    = os.getenv("DEVICE", "cpu")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def print_config() -> None:
    print("\n" + "=" * 65)
    print("CONFIGURACIÓN DEL SISTEMA")
    print("=" * 65)
    print(f"  Ticker      : {TICKER}  [{START_DATE} → {END_DATE}]")
    print(f"  Lookback    : {LOOKBACK} días | t+1 y t+5")
    print(f"  Precio feat : {PRICE_INPUT_SIZE} | LSTM hidden: {LSTM_HIDDEN_SIZE}")
    print(f"  d_model     : {D_MODEL} | num_heads: {NUM_HEADS}")
    print(f"  Batch       : {BATCH_SIZE} | LR: {LEARNING_RATE} | Épocas: {NUM_EPOCHS}")
    print(f"  Walk-fwd    : {WF_SPLITS} splits | test frac: {TEST_FRAC}")
    print(f"  GAN épocas  : {GAN_EPOCHS} | n_critic: {N_CRITIC} | λ_gp: {LAMBDA_GP}")
    print(f"  Sentimiento : FinBERT dim={SENTIMENT_DIM}")
    print(f"  Device      : {DEVICE}")
    print("=" * 65 + "\n")
