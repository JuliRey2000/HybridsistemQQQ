"""
Entrenamiento del Módulo Generativo — TimeGAN + WGAN-GP.

Flujo:
  1. Carga secuencias reales de 20 días desde data/processed/
  2. Entrena TimeGANGenerator + WassersteinCritic con WGAN-GP
  3. Evalúa: Wasserstein Distance, Stylized Facts
  4. Genera escenarios de stress-test condicionados al sentimiento

Ejecutar: python run_train_generative.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

from src.models import TimeGANGenerator, WassersteinCritic, print_model_summary
from src.train import GANTrainer, generate_scenarios
from src.utils import (
    generative_metrics, plot_generated_scenarios,
)
from config import (
    NOISE_DIM, GAN_HIDDEN, GAN_SEQ_LEN, SENTIMENT_DIM,
    GAN_EPOCHS, N_CRITIC, LAMBDA_GP, LR_GEN, LR_CRITIC,
    BATCH_SIZE, N_SCENARIOS,
    DEVICE, LOG_LEVEL, MODELS_PATH, RESULTS_PATH, DATA_PROCESSED_PATH,
    print_config,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/train_generative.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def build_gan_sequences(y_t1: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Construye ventanas deslizantes de 20 días de retornos reales.

    Shape de salida: (n_windows, 20, 1)
    """
    seqs = []
    for i in range(len(y_t1) - window):
        seqs.append(y_t1[i:i+window])
    arr = np.array(seqs, dtype=np.float32)[:, :, np.newaxis]
    logger.info(f"Secuencias GAN construidas: {arr.shape}")
    return arr


def build_sentiment_for_gan(sentiments: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Para cada ventana de 20 días, toma el embedding del primer día como condicionante.
    """
    return sentiments[:len(sentiments) - window]


def main() -> None:
    print_config()
    logger.info("FASE 5 — ENTRENAMIENTO MÓDULO GENERATIVO (TimeGAN + WGAN-GP)")

    # Cargar datos
    p = DATA_PROCESSED_PATH
    y_t1       = np.load(p / "y_t1.npy")
    sentiments = np.load(p / "sentiments.npy")

    real_seqs     = build_gan_sequences(y_t1, window=GAN_SEQ_LEN)
    sent_for_gan  = build_sentiment_for_gan(sentiments, window=GAN_SEQ_LEN)

    # Usar solo train (evitar contaminar con datos de test)
    n_train = int(len(real_seqs) * 0.85)
    real_train = real_seqs[:n_train]
    sent_train = sent_for_gan[:n_train]

    ds = TensorDataset(
        torch.from_numpy(real_train),
        torch.from_numpy(sent_train),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Construir modelos
    generator = TimeGANGenerator(
        noise_dim=NOISE_DIM,
        sentiment_dim=SENTIMENT_DIM,
        hidden_size=GAN_HIDDEN,
        output_seq_len=GAN_SEQ_LEN,
        output_features=1,
    )
    critic = WassersteinCritic(
        seq_features=1,
        sentiment_dim=SENTIMENT_DIM,
        hidden_size=GAN_HIDDEN,
    )

    print_model_summary(generator)
    print_model_summary(critic)

    trainer = GANTrainer(
        generator=generator,
        critic=critic,
        noise_dim=NOISE_DIM,
        device=DEVICE,
        lr_gen=LR_GEN,
        lr_critic=LR_CRITIC,
        n_critic=N_CRITIC,
        lambda_gp=LAMBDA_GP,
    )

    history = trainer.fit(
        real_loader=loader,
        epochs=GAN_EPOCHS,
        log_every=20,
        save_path=str(MODELS_PATH / "timegan_generator.pth"),
    )

    # ─── Evaluación del módulo generativo ─────────────────────────────────
    logger.info("\nEVALUACIÓN MÓDULO GENERATIVO")

    # Generar escenarios con sentimiento del test set
    test_sent = sent_for_gan[n_train:n_train+1]
    fake_seqs = generate_scenarios(
        generator=generator,
        sentiment_emb=test_sent,
        noise_dim=NOISE_DIM,
        n_scenarios=N_SCENARIOS,
        device=DEVICE,
    )

    real_test = real_seqs[n_train:n_train + N_SCENARIOS]
    fake_flat = fake_seqs[:, :, 0]   # (n_scenarios, 20)
    real_flat = real_test[:, :, 0]   # (n_scenarios, 20)

    metrics = generative_metrics(real_flat, fake_flat)
    logger.info("\nMÉTRICAS GENERATIVAS:")
    for k, v in metrics.items():
        if "returns" not in k:
            logger.info(f"  {k:35s}: {v:.4f}")

    plot_generated_scenarios(
        real_flat, fake_flat,
        n_scenarios=15,
        save_path=str(RESULTS_PATH / "generated_scenarios.png"),
    )

    logger.info("\n✅ Módulo generativo completado.")
    logger.info(f"   Wasserstein Distance: {metrics['wasserstein_distance']:.4f}")
    logger.info(f"   Vol. Clustering real/fake: "
                f"{metrics['real_vol_clustering']:.4f} / {metrics['fake_vol_clustering']:.4f}")


if __name__ == "__main__":
    main()
