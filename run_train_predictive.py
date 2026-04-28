"""
Entrenamiento del Módulo Predictivo — Walk-Forward + HybridPredictiveModel.

Flujo:
  1. Carga datos de data/processed/
  2. Reserva test out-of-sample (último 15 %)
  3. Walk-forward de 5 folds sobre el resto
  4. En cada fold: entrena HybridPredictiveModel, evalúa en val
  5. Evaluación final en test: RMSE, DA, Sharpe Ratio

Ejecutar: python run_train_predictive.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.models import HybridPredictiveModel, print_model_summary
from src.train import Trainer, make_dataloader, predict
from src.utils import (
    walk_forward_splits, final_test_split,
    predictive_metrics, sharpe_ratio, long_short_strategy,
    plot_predictions, plot_training_history, plot_cumulative_returns,
)
from config import (
    PRICE_INPUT_SIZE, SENTIMENT_DIM,
    LSTM_HIDDEN_SIZE, D_MODEL, NUM_HEADS, LSTM_NUM_LAYERS, DROPOUT,
    W_T1, W_T5, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, PATIENCE, WF_SPLITS, WF_TRAIN_MIN, TEST_FRAC,
    DEVICE, LOG_LEVEL, MODELS_PATH, RESULTS_PATH, DATA_PROCESSED_PATH,
    print_config,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/train_predictive.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_data() -> dict:
    p = DATA_PROCESSED_PATH
    return {
        "price_seqs": np.load(p / "price_seqs.npy"),
        "sentiments": np.load(p / "sentiments.npy"),
        "y_t1":       np.load(p / "y_t1.npy"),
        "y_t5":       np.load(p / "y_t5.npy"),
    }


def build_model() -> HybridPredictiveModel:
    return HybridPredictiveModel(
        price_input_size=PRICE_INPUT_SIZE,
        sentiment_dim=SENTIMENT_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_lstm_layers=LSTM_NUM_LAYERS,
        dropout=DROPOUT,
    )


def main() -> None:
    print_config()
    logger.info("FASE 3 — ENTRENAMIENTO MÓDULO PREDICTIVO")

    data = load_data()
    n = len(data["y_t1"])

    # Reservar test out-of-sample (cronológicamente al final)
    test_start, _ = final_test_split(n, test_frac=TEST_FRAC)
    train_val_end  = test_start
    logger.info(f"Total muestras: {n} | Train+Val: [0..{train_val_end}) | Test: [{test_start}..{n})")

    # Walk-forward sobre el conjunto train+val
    splits = walk_forward_splits(train_val_end, n_splits=WF_SPLITS, train_min_frac=WF_TRAIN_MIN)

    fold_val_metrics = []
    best_val_loss    = float("inf")
    best_fold        = -1

    for fold, (tr_idx, vl_idx) in enumerate(splits, start=1):
        logger.info(f"\n{'─'*55}")
        logger.info(f"FOLD {fold}/{len(splits)} | Train [{tr_idx[0]}..{tr_idx[-1]}] | Val [{vl_idx[0]}..{vl_idx[-1]}]")
        logger.info(f"{'─'*55}")

        model = build_model()
        if fold == 1:
            print_model_summary(model)

        trainer = Trainer(
            model=model,
            device=DEVICE,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            w_t1=W_T1,
            w_t5=W_T5,
        )

        train_loader = make_dataloader(
            data["price_seqs"], data["sentiments"],
            data["y_t1"], data["y_t5"],
            tr_idx, batch_size=BATCH_SIZE,
        )
        val_loader = make_dataloader(
            data["price_seqs"], data["sentiments"],
            data["y_t1"], data["y_t5"],
            vl_idx, batch_size=BATCH_SIZE,
        )

        save_path = str(MODELS_PATH / f"predictive_fold{fold}.pth")
        history = trainer.fit(
            train_loader, val_loader,
            epochs=NUM_EPOCHS, patience=PATIENCE,
            save_path=save_path,
        )

        # Evaluar en validación de este fold
        preds_t1, preds_t5 = predict(model, data["price_seqs"][vl_idx], data["sentiments"][vl_idx], device=DEVICE)
        m = predictive_metrics(data["y_t1"][vl_idx], preds_t1.flatten())
        m["Sharpe"] = sharpe_ratio(data["y_t1"][vl_idx] * np.sign(preds_t1.flatten()))
        logger.info(f"Fold {fold} val → RMSE: {m['RMSE']:.4f} | DA: {m['Directional_Accuracy']:.3f} | Sharpe: {m['Sharpe']:.3f}")
        fold_val_metrics.append(m)

        # Mantener checkpoint del mejor fold (menor RMSE en validación)
        fold_rmse = m["RMSE"]
        if fold_rmse < best_val_loss:
            best_val_loss = fold_rmse
            best_fold     = fold
            import shutil
            shutil.copy(save_path, str(MODELS_PATH / "hybrid_best.pth"))
            logger.info(f"  → Nuevo mejor fold={fold}  RMSE_val={fold_rmse:.4f}  (guardado como hybrid_best.pth)")

        plot_training_history(history, save_path=str(RESULTS_PATH / f"history_fold{fold}.png"))

    # Resumen walk-forward
    avg_rmse = np.mean([m["RMSE"] for m in fold_val_metrics])
    avg_da   = np.mean([m["Directional_Accuracy"] for m in fold_val_metrics])
    avg_sh   = np.mean([m["Sharpe"] for m in fold_val_metrics])
    logger.info(f"\n{'='*55}")
    logger.info("RESUMEN WALK-FORWARD")
    logger.info(f"  RMSE medio  : {avg_rmse:.4f}")
    logger.info(f"  DA media    : {avg_da:.3f}")
    logger.info(f"  Sharpe medio: {avg_sh:.3f}")
    logger.info(f"  Mejor fold  : {best_fold}  (RMSE_val={best_val_loss:.4f})")
    logger.info(f"{'='*55}")

    # ─── Evaluación final en test out-of-sample ───────────────────────────
    logger.info("\nEVALUACIÓN FINAL — TEST OUT-OF-SAMPLE")

    # Cargar el mejor checkpoint (menor RMSE de validación entre todos los folds)
    best_model = build_model()
    best_model.load_state_dict(torch.load(str(MODELS_PATH / "hybrid_best.pth"), map_location=DEVICE))

    test_idx = np.arange(test_start, n)
    preds_t1_test, preds_t5_test = predict(
        best_model, data["price_seqs"][test_idx], data["sentiments"][test_idx], device=DEVICE
    )

    y_test_t1 = data["y_t1"][test_idx]
    y_test_t5 = data["y_t5"][test_idx]

    m_test_t1 = predictive_metrics(y_test_t1, preds_t1_test.flatten())
    m_test_t5 = predictive_metrics(y_test_t5, preds_t5_test.flatten())

    bt = long_short_strategy(y_test_t1, preds_t1_test.flatten())

    logger.info(f"\nTest t+1 → RMSE: {m_test_t1['RMSE']:.4f} | DA: {m_test_t1['Directional_Accuracy']:.3f}")
    logger.info(f"Test t+5 → RMSE: {m_test_t5['RMSE']:.4f} | DA: {m_test_t5['Directional_Accuracy']:.3f}")
    logger.info(f"Backtesting → Sharpe: {bt['strategy_sharpe']:.3f} | Max DD: {bt['strategy_max_drawdown']:.2f}%")
    logger.info(f"Buy & Hold  → Sharpe: {bt['bh_sharpe']:.3f}")

    plot_predictions(y_test_t1, preds_t1_test.flatten(), horizon="t+1",
                     save_path=str(RESULTS_PATH / "test_predictions_t1.png"))
    plot_cumulative_returns(bt["strategy_returns"], bt["bh_returns"],
                            save_path=str(RESULTS_PATH / "cumulative_returns.png"))

    logger.info("\n✅ Entrenamiento y evaluación completados.")


if __name__ == "__main__":
    main()
