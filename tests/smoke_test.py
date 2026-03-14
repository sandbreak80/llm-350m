"""
Smoke tests — run locally (CPU) before launching AWS training.
Tests: model init, forward pass, param count, loss, checkpoint save/load, generate.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import tempfile

from src.model.config import ModelConfig
from src.model.model import LLM


def test_model_init():
    cfg = ModelConfig()
    model = LLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    n_no_emb = model.num_params(exclude_embeddings=True)
    print(f"  Total params:      {n:>12,}")
    print(f"  Non-emb params:    {n_no_emb:>12,}")
    print(f"  Estimated (cfg):   {cfg.estimate_params():>12,}")
    assert 300_000_000 < n < 420_000_000, f"Param count {n:,} out of expected 300M-420M range"
    print("  PASS: param count in range")


def test_forward_pass():
    cfg = ModelConfig(n_layers=2, n_embd=128, n_heads=4, n_kv_heads=2, intermediate_size=256, block_size=64)
    model = LLM(cfg)
    model.eval()

    B, T = 2, 32
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits, loss = model(idx, targets)

    assert logits.shape == (B, T, cfg.vocab_size), f"Bad logits shape: {logits.shape}"
    assert loss is not None
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    # Initial loss should be near -log(1/vocab_size) ≈ 10.8
    assert 8.0 < loss.item() < 14.0, f"Initial loss {loss.item():.3f} looks wrong (expected 8-14)"
    print(f"  logits shape: {logits.shape}  loss: {loss.item():.4f}")
    print("  PASS: forward pass")


def test_loss_mask():
    cfg = ModelConfig(n_layers=2, n_embd=128, n_heads=4, n_kv_heads=2, intermediate_size=256, block_size=64)
    model = LLM(cfg)
    model.eval()

    B, T = 2, 32
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    # All-zero mask = nothing to train on → loss should be 0
    mask_zero = torch.zeros(B, T, dtype=torch.long)
    with torch.no_grad():
        _, loss_masked = model(idx, targets, loss_mask=mask_zero)

    # All-one mask = same as no mask
    mask_one = torch.ones(B, T, dtype=torch.long)
    with torch.no_grad():
        _, loss_unmasked = model(idx, targets, loss_mask=mask_one)
        _, loss_no_mask = model(idx, targets)

    assert torch.isnan(loss_masked) or loss_masked.item() == 0.0, \
        f"Zero mask should give 0 or NaN loss, got {loss_masked.item()}"
    assert abs(loss_unmasked.item() - loss_no_mask.item()) < 1e-4, \
        f"All-ones mask should match no-mask: {loss_unmasked.item()} vs {loss_no_mask.item()}"
    print(f"  loss(zero mask)={loss_masked.item():.4f}  loss(all-ones)={loss_unmasked.item():.4f}  loss(no mask)={loss_no_mask.item():.4f}")
    print("  PASS: loss masking")


def test_generate():
    cfg = ModelConfig(n_layers=2, n_embd=128, n_heads=4, n_kv_heads=2, intermediate_size=256, block_size=64)
    model = LLM(cfg)
    model.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10)

    assert out.shape == (1, 15), f"Expected (1, 15), got {out.shape}"
    print(f"  Generated shape: {out.shape}")
    print("  PASS: generation")


def test_checkpoint():
    cfg = ModelConfig(n_layers=2, n_embd=128, n_heads=4, n_kv_heads=2, intermediate_size=256, block_size=64)
    model = LLM(cfg)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    # Save
    torch.save({"model": model.state_dict(), "model_config": cfg, "iter": 42, "val_loss": 3.14}, path)

    # Load into fresh model
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model2 = LLM(ckpt["model_config"])
    model2.load_state_dict(ckpt["model"])

    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"

    os.unlink(path)
    print(f"  Checkpoint iter={ckpt['iter']}  val_loss={ckpt['val_loss']}")
    print("  PASS: checkpoint save/load")


def test_training_step():
    """Verify loss decreases after a gradient step."""
    cfg = ModelConfig(n_layers=2, n_embd=128, n_heads=4, n_kv_heads=2, intermediate_size=256, block_size=64)
    model = LLM(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    idx = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        _, loss = model(idx, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  Losses over 5 steps: {[f'{l:.4f}' for l in losses]}")
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    print("  PASS: loss decreases with gradient steps")


if __name__ == "__main__":
    tests = [
        ("Model init + param count", test_model_init),
        ("Forward pass", test_forward_pass),
        ("Loss masking", test_loss_mask),
        ("Generation", test_generate),
        ("Checkpoint save/load", test_checkpoint),
        ("Training step (loss decreases)", test_training_step),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{passed+failed} passed")
    if failed:
        print("SOME TESTS FAILED — do not launch training run")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED — safe to proceed")
