"""
LLM Architecture — Modern 350M parameter model.

Improvements over GPT-2/nanoGPT style (Apex-1 reference):
  - RMSNorm (pre-norm) instead of LayerNorm (post-norm)
  - RoPE positional embeddings instead of learned absolute positions
  - SwiGLU activation instead of GELU
  - Grouped Query Attention (GQA) instead of MHA
  - Flash Attention 2 via PyTorch SDPA
  - No positional embedding table (RoPE is computed on-the-fly)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin cache for a given sequence length."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex
    cos = freqs_cis.real.to(dtype)
    sin = freqs_cis.imag.to(dtype)
    return cos, sin  # each: (seq_len, head_dim // 2)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to query or key tensor.

    Args:
        x: (B, n_heads, T, head_dim)
        cos, sin: (T, head_dim // 2)
    """
    B, H, T, D = x.shape
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * torch.cat([cos, cos], dim=-1) + rotated * torch.cat([sin, sin], dim=-1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_query_groups = config.n_query_groups

        # Projections — no bias (modern practice)
        self.q_proj = nn.Linear(config.n_embd, config.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.n_embd, bias=config.bias)

        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads to match query heads (GQA)
        if self.n_query_groups > 1:
            k = k.repeat_interleave(self.n_query_groups, dim=1)
            v = v.repeat_interleave(self.n_query_groups, dim=1)

        # Flash Attention via PyTorch SDPA (uses FlashAttention-2 kernel when available)
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (as in LLaMA/Mistral)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie token embedding weights to lm_head (reduces params, improves training)
        self.lm_head.weight = self.token_emb.weight

        # RoPE cache (registered as buffer — moves to device with model)
        cos, sin = build_rope_cache(
            config.block_size, config.head_dim, torch.device("cpu"), torch.float32, config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projections by 1/sqrt(2 * n_layers) — GPT-2 paper
        scale = (2 * self.config.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        x = self.emb_dropout(self.token_emb(idx))

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)

            if loss_mask is not None:
                # Instruction finetuning: only compute loss on response tokens
                flat_targets = flat_targets.clone()
                flat_targets[loss_mask.view(-1) == 0] = -100  # ignore prompt tokens

            loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.65,
        top_k: int = 25,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def num_params(self, exclude_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.token_emb.weight.numel()
        return n
