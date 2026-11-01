"""Validated, serializable settings shared by all paper experiments."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


VARIANTS = (
    "sequence_flat", "position_pool", "transformer_single", "transformer_multi",
    "madp_transformer", "din_single", "madp_din", "pal", "dpin",
)


@dataclass(frozen=True)
class Config:
    variant: str = "madp_transformer"
    vocab_size: int = 1001
    context_dim: int = 8
    num_positions: int = 20
    history_length: int = 20
    embedding_dim: int = 16
    model_dim: int = 32
    num_heads: int = 4
    num_layers: int = 1
    ffn_dim: int = 64
    mlp_dims: tuple = (64, 32)
    dropout: float = 0.1
    adversarial_weight: float = 0.1
    grl_strength: float = 1.0
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 5
    seed: int = 42

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        for name in ("vocab_size", "context_dim", "num_positions", "history_length",
                     "embedding_dim", "model_dim", "num_heads", "num_layers",
                     "ffn_dim", "batch_size", "epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab_size < 2 or self.num_positions < 2:
            raise ValueError("vocab_size and num_positions must be at least 2")
        if self.vocab_size > 2**31 or self.num_positions > 2**31 - 1:
            raise ValueError("item and position IDs must fit in int32")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        object.__setattr__(self, "mlp_dims", tuple(self.mlp_dims))
        if not self.mlp_dims or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
                                    for x in self.mlp_dims):
            raise ValueError("mlp_dims must contain positive integers")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        for name in ("adversarial_weight", "grl_strength", "learning_rate"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")

    @property
    def raw_history_length(self):
        return self.num_positions * self.history_length

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**values)

