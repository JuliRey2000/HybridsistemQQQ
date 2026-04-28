"""
Entrenamiento de los dos módulos del sistema.

MÓDULO PREDICTIVO:
  - Trainer: entrena HybridPredictiveModel con pérdida multi-step (t+1, t+5)
  - Walk-forward validation cronológica (sin data leakage)
  - Early stopping sobre val_loss combinada

MÓDULO GENERATIVO:
  - GANTrainer: entrena TimeGANGenerator + WassersteinCritic
  - WGAN-GP: reemplaza weight clipping por Gradient Penalty para estabilidad
  - Ratio de actualización n_critic : n_gen (default 5:1)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ============================================================================
# DATASET HELPERS
# ============================================================================

def make_dataloader(
    price_seqs: np.ndarray,
    sentiments: np.ndarray,
    targets_t1: np.ndarray,
    targets_t5: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """
    Crea un DataLoader a partir de arrays numpy.

    IMPORTANTE: shuffle=False por defecto para preservar orden temporal.
    """
    ds = TensorDataset(
        torch.from_numpy(price_seqs[indices]).float(),
        torch.from_numpy(sentiments[indices]).float(),
        torch.from_numpy(targets_t1[indices]).float().unsqueeze(1),
        torch.from_numpy(targets_t5[indices]).float().unsqueeze(1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                logger.info("Early stopping activado.")


# ============================================================================
# MÓDULO PREDICTIVO — Trainer
# ============================================================================

class Trainer:
    """
    Entrena HybridPredictiveModel con pérdida Huber multi-step.

    La pérdida total combina t+1 y t+5 con peso ajustable:
        L = w1 * Huber(pred_t1, y_t1) + w5 * Huber(pred_t5, y_t5)

    Huber es más robusta que MSE ante retornos extremos (e.g. COVID crash).

    Args:
        model      : instancia de HybridPredictiveModel
        device     : 'cpu' o 'cuda'
        lr         : learning rate (default: 1e-3)
        weight_decay: L2 regularización (default: 1e-5)
        w_t1       : peso de la pérdida t+1 (default: 0.6)
        w_t5       : peso de la pérdida t+5 (default: 0.4)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        w_t1: float = 0.6,
        w_t5: float = 0.4,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.w_t1 = w_t1
        self.w_t5 = w_t5

        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Scheduler: reduce LR si la validación no mejora en 5 épocas
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5, verbose=False
        )

        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_da": [], "val_da": [],
        }

    def _step(
        self,
        loader: DataLoader,
        training: bool = True,
    ) -> tuple[float, float]:
        """Un paso de entrenamiento o validación."""
        self.model.train(training)
        total_loss, total_da, n = 0.0, 0.0, 0

        with torch.set_grad_enabled(training):
            for price_seq, sentiment, y_t1, y_t5 in loader:
                price_seq  = price_seq.to(self.device)
                sentiment  = sentiment.to(self.device)
                y_t1       = y_t1.to(self.device)
                y_t5       = y_t5.to(self.device)

                pred_t1, pred_t5 = self.model(price_seq, sentiment)

                loss = self.w_t1 * self.criterion(pred_t1, y_t1) \
                     + self.w_t5 * self.criterion(pred_t5, y_t5)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                # Directional accuracy sobre t+1
                da = (torch.sign(pred_t1) == torch.sign(y_t1)).float().mean().item()

                bs = price_seq.size(0)
                total_loss += loss.item() * bs
                total_da   += da * bs
                n          += bs

        return total_loss / n, total_da / n

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
        save_path: str = "models/predictive_best.pth",
    ) -> dict:
        """
        Bucle de entrenamiento.

        Returns:
            history dict con train/val loss y directional accuracy por época
        """
        early = EarlyStopping(patience=patience)
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            tr_loss, tr_da = self._step(train_loader, training=True)
            vl_loss, vl_da = self._step(val_loader,   training=False)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["train_da"].append(tr_da)
            self.history["val_da"].append(vl_da)

            self.scheduler.step(vl_loss)

            if vl_loss < best_val:
                best_val = vl_loss
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), save_path)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Época {epoch:3d}/{epochs} | "
                    f"Train loss: {tr_loss:.6f} | Val loss: {vl_loss:.6f} | "
                    f"Val DA: {vl_da:.3f}"
                )

            early(vl_loss)
            if early.stop:
                logger.info(f"Detenido en época {epoch}.")
                break

        logger.info(f"Mejor val_loss: {best_val:.6f} — modelo en {save_path}")
        return self.history


def predict(
    model: nn.Module,
    price_seqs: np.ndarray,
    sentiments: np.ndarray,
    device: str = "cpu",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inferencia en batch sobre arrays numpy.

    Returns:
        (preds_t1, preds_t5): arrays numpy (n_samples, 1)
    """
    model.eval()
    dev = torch.device(device)
    preds_t1, preds_t5 = [], []

    with torch.no_grad():
        for i in range(0, len(price_seqs), batch_size):
            ps = torch.from_numpy(price_seqs[i:i+batch_size]).float().to(dev)
            se = torch.from_numpy(sentiments[i:i+batch_size]).float().to(dev)
            p1, p5 = model(ps, se)
            preds_t1.append(p1.cpu().numpy())
            preds_t5.append(p5.cpu().numpy())

    return np.concatenate(preds_t1), np.concatenate(preds_t5)


# ============================================================================
# MÓDULO GENERATIVO — WGAN-GP Trainer
# ============================================================================

def _gradient_penalty(
    critic: nn.Module,
    real_seq: torch.Tensor,
    fake_seq: torch.Tensor,
    sentiment: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Calcula la Gradient Penalty de WGAN-GP.

    La GP penaliza al crítico cuando la norma del gradiente respecto a una
    interpolación entre datos reales y falsos se aleja de 1.
    Esto fuerza al crítico a ser 1-Lipschitz sin necesidad de recortar pesos.

    Ecuación:
        GP = E[(||∇_x̂ D(x̂)||_2 - 1)²]
    donde x̂ = ε·real + (1-ε)·fake, ε ~ Uniform[0,1]

    Args:
        critic    : WassersteinCritic
        real_seq  : (batch, seq_len, features) — secuencias reales
        fake_seq  : (batch, seq_len, features) — secuencias generadas
        sentiment : (batch, sentiment_dim) — embedding condicionante
        device    : dispositivo de cómputo
        lambda_gp : coeficiente de penalización (default: 10)
    """
    batch = real_seq.size(0)
    eps = torch.rand(batch, 1, 1, device=device)
    eps = eps.expand_as(real_seq)

    interpolated = (eps * real_seq + (1 - eps) * fake_seq).requires_grad_(True)

    score_interp = critic(interpolated, sentiment)

    gradients = torch.autograd.grad(
        outputs=score_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # (batch, seq_len, features)

    grad_norm = gradients.view(batch, -1).norm(2, dim=1)   # (batch,)
    gp = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
    return gp


class GANTrainer:
    """
    Entrena TimeGANGenerator + WassersteinCritic con WGAN-GP.

    Flujo por paso:
      1. Actualizar crítico `n_critic` veces:
           L_critic = E[D(fake)] - E[D(real)] + GP
      2. Actualizar generador 1 vez:
           L_gen = -E[D(G(z, sentimiento))]

    El ratio n_critic : 1 (default 5:1) asegura que el crítico
    converja antes de actualizar el generador, evitando el colapso de modo.

    Args:
        generator   : TimeGANGenerator
        critic      : WassersteinCritic
        noise_dim   : dimensión del vector de ruido z
        device      : 'cpu' o 'cuda'
        lr_gen      : learning rate generador (default: 1e-4)
        lr_critic   : learning rate crítico (default: 1e-4)
        n_critic    : pasos de crítico por paso de generador (default: 5)
        lambda_gp   : coeficiente de gradient penalty (default: 10)
    """

    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        noise_dim: int = 32,
        device: str = "cpu",
        lr_gen: float = 1e-4,
        lr_critic: float = 1e-4,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
    ):
        self.gen = generator.to(device)
        self.critic = critic.to(device)
        self.noise_dim = noise_dim
        self.device = torch.device(device)
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        # Adam con β1=0 es estándar en literatura WGAN-GP
        self.opt_gen    = optim.Adam(generator.parameters(), lr=lr_gen,    betas=(0.0, 0.9))
        self.opt_critic = optim.Adam(critic.parameters(),    lr=lr_critic, betas=(0.0, 0.9))

        self.history: dict[str, list] = {"critic_loss": [], "gen_loss": [], "w_distance": []}

    def _sample_noise(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.noise_dim, device=self.device)

    def train_epoch(self, real_loader: DataLoader) -> dict:
        """
        Entrena una época completa sobre el DataLoader de secuencias reales.

        El loader devuelve (real_seq, sentiment_emb) por batch.

        Returns:
            dict con pérdidas promedio del época
        """
        self.gen.train()
        self.critic.train()

        epoch_critic, epoch_gen, epoch_wd = [], [], []
        data_iter = iter(real_loader)

        while True:
            # === n_critic pasos del crítico ===
            critic_losses = []
            for _ in range(self.n_critic):
                try:
                    real_seq, sentiment = next(data_iter)
                except StopIteration:
                    data_iter = iter(real_loader)
                    real_seq, sentiment = next(data_iter)

                real_seq  = real_seq.to(self.device)
                sentiment = sentiment.to(self.device)
                batch = real_seq.size(0)

                z = self._sample_noise(batch)
                with torch.no_grad():
                    fake_seq = self.gen(z, sentiment)

                score_real = self.critic(real_seq, sentiment)
                score_fake = self.critic(fake_seq.detach(), sentiment)

                gp = _gradient_penalty(
                    self.critic, real_seq, fake_seq.detach(),
                    sentiment, self.device, self.lambda_gp
                )

                loss_critic = score_fake.mean() - score_real.mean() + gp

                self.opt_critic.zero_grad()
                loss_critic.backward()
                self.opt_critic.step()

                # Wasserstein distance aproximada (sin GP)
                w_dist = (score_real.mean() - score_fake.mean()).item()
                critic_losses.append(loss_critic.item())
                epoch_wd.append(w_dist)

            epoch_critic.extend(critic_losses)

            # === 1 paso del generador ===
            try:
                real_seq, sentiment = next(data_iter)
            except StopIteration:
                break  # fin de época

            real_seq  = real_seq.to(self.device)
            sentiment = sentiment.to(self.device)
            batch = real_seq.size(0)

            z = self._sample_noise(batch)
            fake_seq = self.gen(z, sentiment)
            score_fake = self.critic(fake_seq, sentiment)

            loss_gen = -score_fake.mean()

            self.opt_gen.zero_grad()
            loss_gen.backward()
            self.opt_gen.step()

            epoch_gen.append(loss_gen.item())

        return {
            "critic_loss": float(np.mean(epoch_critic)) if epoch_critic else 0.0,
            "gen_loss":    float(np.mean(epoch_gen))    if epoch_gen    else 0.0,
            "w_distance":  float(np.mean(epoch_wd))     if epoch_wd     else 0.0,
        }

    def fit(
        self,
        real_loader: DataLoader,
        epochs: int = 200,
        log_every: int = 20,
        save_path: str = "models/timegan_generator.pth",
    ) -> dict:
        """
        Bucle de entrenamiento GAN completo.

        Returns:
            history dict
        """
        logger.info(f"Iniciando WGAN-GP: {epochs} épocas, n_critic={self.n_critic}")

        for epoch in range(1, epochs + 1):
            stats = self.train_epoch(real_loader)
            self.history["critic_loss"].append(stats["critic_loss"])
            self.history["gen_loss"].append(stats["gen_loss"])
            self.history["w_distance"].append(stats["w_distance"])

            if epoch % log_every == 0 or epoch == 1:
                logger.info(
                    f"Época {epoch:4d}/{epochs} | "
                    f"Critic: {stats['critic_loss']:+.4f} | "
                    f"Gen: {stats['gen_loss']:+.4f} | "
                    f"W-dist≈{stats['w_distance']:.4f}"
                )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.gen.state_dict(), save_path)
        logger.info(f"Generador guardado en {save_path}")
        return self.history


def generate_scenarios(
    generator: nn.Module,
    sentiment_emb: np.ndarray,
    noise_dim: int = 32,
    n_scenarios: int = 100,
    device: str = "cpu",
) -> np.ndarray:
    """
    Genera escenarios sintéticos de 20 días condicionados al sentimiento.

    Args:
        generator     : TimeGANGenerator entrenado
        sentiment_emb : (1, 768) o (n, 768) — embedding de sentimiento
        noise_dim     : dimensión del ruido z
        n_scenarios   : cuántos escenarios generar por embedding
        device        : dispositivo

    Returns:
        scenarios: (n_scenarios, 20, features) — trayectorias generadas
    """
    generator.eval()
    dev = torch.device(device)

    if sentiment_emb.ndim == 1:
        sentiment_emb = sentiment_emb[np.newaxis, :]

    sent_t = torch.from_numpy(
        np.repeat(sentiment_emb, n_scenarios, axis=0)
    ).float().to(dev)

    z = torch.randn(n_scenarios, noise_dim, device=dev)

    with torch.no_grad():
        scenarios = generator(z, sent_t).cpu().numpy()

    return scenarios  # (n_scenarios, 20, features)
