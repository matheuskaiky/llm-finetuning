"""Interfaces shared by the task modules (Q1-Q6).

Each task is implemented as a subclass behind one of these abstractions and
selected through configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# A loaded model paired with its tokenizer, as returned by a ModelProvider.
ModelBundle = tuple[Any, Any]


@dataclass
class TrainResult:
    """Outcome of a training run.

    Attributes:
        model: The trained model object.
        metrics: Scalar metrics collected during/after training.
        output_dir: Where checkpoints/artifacts were written, if any.
    """

    model: Any
    metrics: dict[str, float] = field(default_factory=dict)
    output_dir: str | None = None


class ModelProvider(ABC):
    """Loads a model and its tokenizer. Implementations vary by backend
    (e.g. local weights vs. cloud endpoint)."""

    @abstractmethod
    def load(self) -> ModelBundle:
        """Return a ``(model, tokenizer)`` bundle."""


class DatasetLoader(ABC):
    """Produces a dataset from some raw source (files, PDFs, generated pairs)."""

    @abstractmethod
    def load(self) -> Any:
        """Return the loaded/processed dataset."""


class Trainer(ABC):
    """A training strategy (continual pretrain, SFT, LoRA/QLoRA, distillation)."""

    @abstractmethod
    def train(self, model: Any, dataset: Any, config: Any) -> TrainResult:
        """Train ``model`` on ``dataset`` and return a :class:`TrainResult`."""


class Metric(ABC):
    """An evaluation metric accumulated over batches via update/compute/reset."""

    #: Stable identifier used in result dicts and the registry.
    name: str = "metric"

    @abstractmethod
    def update(self, logits: Any, labels: Any) -> None:
        """Accumulate statistics from a batch of ``logits`` and ``labels``."""

    @abstractmethod
    def compute(self) -> float:
        """Return the final metric value over everything seen so far."""

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state."""


class Evaluator(ABC):
    """Runs a model over a benchmark and returns a dict of metric values."""

    @abstractmethod
    def evaluate(self, model: Any, benchmark: Any) -> dict[str, float]:
        """Evaluate ``model`` on ``benchmark`` and return ``{metric: value}``."""
