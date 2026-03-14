from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data"

    # Optimization
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 2000
    lr_decay_iters: int = 60000
    max_iters: int = 60000
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Batch
    batch_size: int = 4
    gradient_accumulation_steps: int = 32  # effective batch = 4 * 32 = 128

    # Logging & checkpointing
    log_interval: int = 100
    eval_interval: int = 1000
    eval_iters: int = 100
    checkpoint_interval: int = 500          # Save every N iters for spot resilience
    checkpoint_dir: str = "checkpoints"

    # W&B (set to None to disable)
    wandb_project: str = "llm-350m-pretrain"

    # System
    compile: bool = True
    seed: int = 42
