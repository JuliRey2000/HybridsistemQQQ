"""
Arquitecturas del Sistema Híbrido de Deep Learning para predicción QQQ.

Módulos implementados según tesis:

MÓDULO PREDICTIVO (LSTM + Cross-Attention):
  - SelfAttentionLayer   : Atención sobre la secuencia temporal
  - LSTMWithAttention    : LSTM + Self-Attention encoder
  - CrossAttentionFusion : Fusión intermedia entre OHLCV y FinBERT embeddings
  - HybridPredictiveModel: Predicción multi-step t+1 y t+5

MÓDULO GENERATIVO (TimeGAN condicional + WGAN-GP):
  - TimeGANGenerator     : Generador de trayectorias sintéticas (20 días)
  - WassersteinCritic    : Crítico con soporte para gradiente de Wasserstein

Referencia arquitectónica:
  Yoon et al. (2019) TimeGAN + adaptación CGAN + WGAN-GP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MÓDULO PREDICTIVO
# ============================================================================

class SelfAttentionLayer(nn.Module):
    """
    Capa de Self-Attention sobre secuencia temporal.

    Permite al modelo ponderar qué pasos de tiempo son más relevantes
    para la predicción final. Por ejemplo: si hubo alta volatilidad hace
    5 días, el modelo aprende a prestarle más atención.

    Args:
        hidden_dim (int): Dimensión de la representación interna
        num_heads (int): Número de cabezas de atención (default: 4)
        dropout (float): Dropout sobre los pesos de atención (default: 0.1)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) debe ser divisible por num_heads ({num_heads})"
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            out: (batch, seq_len, hidden_dim) — secuencia con contexto global
        """
        attn_out, _ = self.attn(x, x, x)
        out = self.norm(x + self.dropout(attn_out))  # residual connection
        return out


class LSTMWithAttention(nn.Module):
    """
    Encoder de series temporales: LSTM + Self-Attention.

    Flujo:
      Input OHLCV+indicadores → LSTM (captura dependencias temporales)
                              → Self-Attention (pondera pasos relevantes)
                              → Vector contexto de dimensión `d_model`

    Args:
        input_size (int): Número de features por paso de tiempo (ej: 14)
        hidden_size (int): Unidades ocultas LSTM (default: 128)
        num_layers (int): Capas LSTM apiladas (default: 2)
        d_model (int): Dimensión del embedding de salida (default: 64)
        num_heads (int): Cabezas de atención (default: 4)
        dropout (float): Tasa de dropout (default: 0.2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # unidireccional: solo usamos historia pasada
        )

        # Proyectar la salida LSTM a d_model antes de attention
        self.proj = nn.Linear(hidden_size, d_model)

        self.attention = SelfAttentionLayer(
            hidden_dim=d_model, num_heads=num_heads, dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            context: (batch, d_model) — embedding comprimido de la secuencia
        """
        lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden_size)
        projected = self.proj(lstm_out)         # (batch, seq_len, d_model)
        attended = self.attention(projected)    # (batch, seq_len, d_model)

        # Promedio ponderado sobre la secuencia como vector final
        context = attended.mean(dim=1)          # (batch, d_model)
        context = self.norm(self.dropout(context))
        return context


class CrossAttentionFusion(nn.Module):
    """
    Fusión Intermedia con Cross-Attention entre series temporal y sentimiento.

    A diferencia de la concatenación simple, aquí la secuencia temporal
    'consulta' el embedding de sentimiento para modular qué aspectos
    de la historia de precios son más relevantes dado el estado emocional
    del mercado en ese día.

    Query  = embedding de precios (lo que queremos enriquecer)
    Key/Value = embedding de sentimiento (contexto que aporta información)

    Args:
        d_model (int): Dimensión de ambos embeddings (deben ser iguales)
        num_heads (int): Cabezas de atención (default: 4)
        dropout (float): Dropout (default: 0.1)
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, price_emb: torch.Tensor, sentiment_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            price_emb    : (batch, d_model) — embedding de precios
            sentiment_emb: (batch, d_model) — embedding de sentimiento FinBERT

        Returns:
            fused: (batch, d_model) — embedding fusionado
        """
        # Expandir a forma (batch, 1, d_model) para MultiheadAttention
        q = price_emb.unsqueeze(1)       # query: señal de precios
        k = sentiment_emb.unsqueeze(1)   # key: contexto de sentimiento
        v = sentiment_emb.unsqueeze(1)   # value: contexto de sentimiento

        attn_out, _ = self.cross_attn(q, k, v)  # (batch, 1, d_model)
        attn_out = attn_out.squeeze(1)            # (batch, d_model)

        # Residual: enriquece el embedding de precios con sentimiento
        fused = self.norm(price_emb + self.dropout(attn_out))
        return fused


class HybridPredictiveModel(nn.Module):
    """
    Modelo Predictivo Híbrido con fusión Cross-Attention y multi-step output.

    Arquitectura completa:
      1. LSTMWithAttention   procesa ventana OHLCV+indicadores (30 días)
      2. FinBERT embedding   recibe vector de sentimiento diario (pre-computado)
      3. CrossAttentionFusion fusiona ambas ramas en un embedding conjunto
      4. Dos cabezas de salida independientes:
           - head_t1: predicción de retorno en t+1
           - head_t5: predicción de retorno en t+5

    La predicción multi-step es directa (direct forecasting), no autoregresiva,
    lo que evita la acumulación de error en horizontes más largos.

    Args:
        price_input_size (int): Features de la secuencia de precios (default: 14)
        sentiment_dim (int): Dimensión del embedding FinBERT (default: 768)
        hidden_size (int): Unidades LSTM (default: 128)
        d_model (int): Dimensión interna (default: 64)
        num_heads (int): Cabezas de atención (default: 4)
        num_lstm_layers (int): Capas LSTM (default: 2)
        dropout (float): Dropout global (default: 0.2)
    """

    def __init__(
        self,
        price_input_size: int = 14,
        sentiment_dim: int = 768,
        hidden_size: int = 128,
        d_model: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Rama 1: LSTM + Self-Attention sobre series de precios
        self.price_encoder = LSTMWithAttention(
            input_size=price_input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Rama 2: Proyección del embedding FinBERT (768 dim → d_model)
        self.sentiment_proj = nn.Sequential(
            nn.Linear(sentiment_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # Fusión intermedia con Cross-Attention
        self.fusion = CrossAttentionFusion(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )

        # Cabeza de predicción compartida (MLP base)
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cabezas independientes por horizonte
        self.head_t1 = nn.Linear(32, 1)   # retorno t+1
        self.head_t5 = nn.Linear(32, 1)   # retorno t+5

    def forward(
        self,
        price_seq: torch.Tensor,
        sentiment_emb: torch.Tensor,
    ):
        """
        Args:
            price_seq    : (batch, seq_len, price_input_size)
            sentiment_emb: (batch, sentiment_dim) — embedding FinBERT del día

        Returns:
            pred_t1: (batch, 1)
            pred_t5: (batch, 1)
        """
        price_ctx = self.price_encoder(price_seq)          # (batch, d_model)
        sent_ctx = self.sentiment_proj(sentiment_emb)      # (batch, d_model)
        fused = self.fusion(price_ctx, sent_ctx)           # (batch, d_model)

        shared = self.shared_head(fused)
        pred_t1 = self.head_t1(shared)                     # (batch, 1)
        pred_t5 = self.head_t5(shared)                     # (batch, 1)

        return pred_t1, pred_t5


# ============================================================================
# MÓDULO GENERATIVO: TimeGAN Condicional + WGAN-GP
# ============================================================================

class TimeGANGenerator(nn.Module):
    """
    Generador de trayectorias sintéticas de 20 días (TimeGAN Condicional).

    Genera secuencias de retornos realistas condicionadas al vector de
    sentimiento del día inicial. Esto permite simular escenarios:
    - Con sentimiento negativo extremo (como COVID crash)
    - Con sentimiento positivo (rally de mercado)

    Arquitectura:
      [ruido z + sentimiento] → LSTM → Linear → retornos sintéticos (20 días)

    La conditioning por sentimiento implementa el enfoque CGAN (Conditional GAN):
    el sentimiento es inyectado en cada paso de la generación.

    Args:
        noise_dim (int): Dimensión del vector de ruido z (default: 32)
        sentiment_dim (int): Dimensión del embedding de sentimiento (default: 768)
        hidden_size (int): Unidades LSTM del generador (default: 128)
        output_seq_len (int): Pasos de tiempo a generar (default: 20)
        output_features (int): Features por paso (ej: retorno diario = 1)
        num_layers (int): Capas LSTM (default: 2)
        dropout (float): Dropout (default: 0.1)
    """

    def __init__(
        self,
        noise_dim: int = 32,
        sentiment_dim: int = 768,
        hidden_size: int = 128,
        output_seq_len: int = 20,
        output_features: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_seq_len = output_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Proyectar sentimiento a espacio del generador
        self.sentiment_proj = nn.Sequential(
            nn.Linear(sentiment_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Combinar ruido + sentimiento como input del LSTM
        lstm_input_dim = noise_dim + 32  # ruido + sentimiento proyectado

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_size, output_features)

    def forward(
        self, z: torch.Tensor, sentiment_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z            : (batch, noise_dim) — ruido aleatorio
            sentiment_emb: (batch, sentiment_dim) — condicionante de sentimiento

        Returns:
            generated: (batch, output_seq_len, output_features)
        """
        sent_proj = self.sentiment_proj(sentiment_emb)  # (batch, 32)

        # Replicar el input condicionado en cada paso de la secuencia
        cond = torch.cat([z, sent_proj], dim=-1)                  # (batch, noise_dim+32)
        cond_seq = cond.unsqueeze(1).repeat(1, self.output_seq_len, 1)  # (batch, 20, input_dim)

        lstm_out, _ = self.lstm(cond_seq)           # (batch, 20, hidden_size)
        generated = self.output_proj(lstm_out)      # (batch, 20, output_features)
        return generated


class WassersteinCritic(nn.Module):
    """
    Crítico Wasserstein para WGAN-GP.

    A diferencia de un discriminador clásico (que clasifica real/falso),
    el crítico Wasserstein produce un score escalar sin sigmoide.
    Mide la distancia de Wasserstein entre la distribución real y la generada.

    La estabilidad matemática se logra con Gradient Penalty (GP) en train.py,
    que reemplaza el weight clipping clásico de WGAN.

    Arquitectura:
      [secuencia + sentimiento] → LSTM → Linear → score real (sin límite)

    Args:
        seq_features (int): Features por paso de tiempo en la secuencia real
        sentiment_dim (int): Dimensión del embedding de sentimiento
        hidden_size (int): Unidades LSTM (default: 128)
        num_layers (int): Capas LSTM (default: 2)
        dropout (float): Dropout (default: 0.1)
    """

    def __init__(
        self,
        seq_features: int = 1,
        sentiment_dim: int = 768,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.sentiment_proj = nn.Sequential(
            nn.Linear(sentiment_dim, 32),
            nn.LeakyReLU(0.2),
        )

        lstm_input_dim = seq_features + 32

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            # Sin sigmoide — el crítico produce scores no acotados
        )

    def forward(
        self, seq: torch.Tensor, sentiment_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            seq          : (batch, seq_len, seq_features) — secuencia real o generada
            sentiment_emb: (batch, sentiment_dim)

        Returns:
            score: (batch, 1) — score de realismo (mayor = más real)
        """
        sent_proj = self.sentiment_proj(sentiment_emb)              # (batch, 32)
        sent_seq = sent_proj.unsqueeze(1).repeat(1, seq.size(1), 1) # (batch, seq_len, 32)

        inp = torch.cat([seq, sent_seq], dim=-1)   # (batch, seq_len, features+32)
        lstm_out, _ = self.lstm(inp)
        last = lstm_out[:, -1, :]                  # (batch, hidden_size)
        score = self.score_head(last)              # (batch, 1)
        return score


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module) -> None:
    print(f"\n{'='*60}")
    print(f"MODELO: {model.__class__.__name__}")
    print(f"{'='*60}")
    print(model)
    print(f"\nParámetros entrenables: {count_parameters(model):,}")
    print(f"{'='*60}\n")


# ============================================================================
# TEST DE FORMAS (verificación rápida)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    B, T, F = 8, 30, 14   # batch, seq_len, price_features
    S = 768               # FinBERT embedding dim

    print("\n=== TEST: HybridPredictiveModel ===")
    model = HybridPredictiveModel(
        price_input_size=F, sentiment_dim=S, d_model=64, num_heads=4
    )
    price_seq = torch.randn(B, T, F)
    sent_emb  = torch.randn(B, S)
    t1, t5 = model(price_seq, sent_emb)
    print(f"pred_t1 shape: {t1.shape}")   # (8, 1)
    print(f"pred_t5 shape: {t5.shape}")   # (8, 1)
    print_model_summary(model)

    print("\n=== TEST: TimeGANGenerator ===")
    gen = TimeGANGenerator(noise_dim=32, sentiment_dim=S, output_seq_len=20)
    z = torch.randn(B, 32)
    gen_seq = gen(z, sent_emb)
    print(f"generated shape: {gen_seq.shape}")  # (8, 20, 1)
    print_model_summary(gen)

    print("\n=== TEST: WassersteinCritic ===")
    critic = WassersteinCritic(seq_features=1, sentiment_dim=S)
    score_real = critic(gen_seq, sent_emb)
    print(f"critic score shape: {score_real.shape}")  # (8, 1)
    print_model_summary(critic)
