"""Language-model evaluator: runs a causal LM over a corpus and reports metrics.

Loads texts from a benchmark file, runs one forward pass per document, applies
the causal shift, and feeds the aligned ``(logits, labels)`` to each configured
metric. ``torch`` is imported lazily inside :meth:`evaluate`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..core.config import EvaluationConfig
from ..core.interfaces import Evaluator, Metric, ModelBundle
from ..core.registry import EVALUATORS, METRICS


def load_benchmark_texts(path: str | Path) -> list[str]:
    """Read benchmark documents from a ``.txt`` (whole file) or ``.jsonl`` file.

    For ``.jsonl`` each line is a JSON object; the ``text`` field is used, or the
    concatenation of ``instruction``/``input``/``output`` when present.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        texts: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                texts.append(obj["text"])
            else:
                parts = [obj.get(k, "") for k in ("instruction", "input", "output")]
                texts.append("\n".join(p for p in parts if p))
        return texts
    # Plain text: treat the whole file as one document.
    return [path.read_text(encoding="utf-8")]


@EVALUATORS.register("language_model")
class LanguageModelEvaluator(Evaluator):
    """Computes intrinsic next-token metrics over a set of documents."""

    name = "language_model"

    def __init__(
        self,
        metrics: list[Metric] | None = None,
        max_length: int = 1024,
        stride: int = 0,
    ) -> None:
        self.metrics = metrics or [
            METRICS.build("perplexity"),
            METRICS.build("cross_entropy"),
            METRICS.build("token_accuracy"),
        ]
        self.max_length = max_length
        self.stride = stride

    @classmethod
    def from_config(cls, config: EvaluationConfig) -> LanguageModelEvaluator:
        """Build an evaluator (metrics + windowing) from an EvaluationConfig."""
        metrics = [METRICS.build(name) for name in config.metrics]
        return cls(metrics=metrics, max_length=config.max_length, stride=config.stride)

    def evaluate(self, model: ModelBundle, benchmark: Any) -> dict[str, float]:
        """Evaluate a ``(model, tokenizer)`` bundle over ``benchmark``.

        ``benchmark`` may be a path to a corpus file or an iterable of strings.
        """
        import torch  # lazy: only needed when actually running a model

        net, tokenizer = model
        net.eval()
        device = next(net.parameters()).device

        texts: Iterable[str]
        if isinstance(benchmark, (str, Path)):
            texts = load_benchmark_texts(benchmark)
        else:
            texts = benchmark

        for metric in self.metrics:
            metric.reset()

        with torch.no_grad():
            for text in texts:
                if not text.strip():
                    continue
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                input_ids = enc["input_ids"].to(device)
                if input_ids.shape[1] < 2:
                    continue  # need at least one (context, target) pair
                logits = net(input_ids).logits
                # Causal shift: predict token t+1 from tokens <= t.
                shift_logits = logits[:, :-1, :].float().cpu().numpy()
                shift_labels = input_ids[:, 1:].cpu().numpy()
                for metric in self.metrics:
                    metric.update(shift_logits, shift_labels)

        return {metric.name: metric.compute() for metric in self.metrics}


def save_results(results: dict[str, float], path: str | Path) -> Path:
    """Persist a metric dict as pretty JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
