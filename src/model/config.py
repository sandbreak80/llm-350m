from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Vocabulary
    vocab_size: int = 50304         # GPT-2 vocab, padded to multiple of 64

    # Architecture
    n_layers: int = 24
    n_heads: int = 16               # Query heads
    n_kv_heads: int = 4             # KV heads for Grouped Query Attention (GQA)
    n_embd: int = 1024
    intermediate_size: int = 2816   # SwiGLU FFN: ~2.75x hidden (accounts for gating)
    block_size: int = 2048          # Context length (2x reference model)

    # Regularization
    dropout: float = 0.0            # 0 for pretraining, 0.1 for finetuning
    bias: bool = False              # No bias: cleaner, faster, slightly better

    # RoPE
    rope_theta: float = 10000.0     # RoPE base frequency

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_heads

    @property
    def n_query_groups(self) -> int:
        """Number of query heads per KV head."""
        return self.n_heads // self.n_kv_heads

    def estimate_params(self) -> int:
        """Rough parameter count estimate."""
        # Embedding
        embed = self.vocab_size * self.n_embd
        # Attention per layer: Q, K, V, O projections
        attn = (
            self.n_embd * self.n_embd  # Q
            + 2 * (self.n_kv_heads * self.head_dim) * self.n_embd  # K, V (GQA)
            + self.n_embd * self.n_embd  # O
        ) * self.n_layers
        # FFN per layer: gate, up, down (SwiGLU — 3 matrices)
        ffn = (3 * self.n_embd * self.intermediate_size) * self.n_layers
        # RMSNorm: 2 per layer + 1 final
        norm = self.n_embd * (2 * self.n_layers + 1)
        # LM head shares embedding weights (tied)
        return embed + attn + ffn + norm
