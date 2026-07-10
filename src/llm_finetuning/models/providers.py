"""Model providers for local weights and (placeholder) cloud endpoints.

``transformers`` and ``torch`` are imported lazily inside :meth:`load` so the
module can be imported without the ML stack installed.
"""

from __future__ import annotations

from typing import Any

from ..core.interfaces import ModelBundle, ModelProvider
from ..core.registry import MODEL_PROVIDERS


@MODEL_PROVIDERS.register("local")
class LocalModelProvider(ModelProvider):
    """Loads a Hugging Face causal LM and tokenizer from local/hub weights.

    Args:
        model_name: Hub id or local path.
        dtype: Torch dtype name (e.g. ``"float16"``, ``"bfloat16"``), or None.
        device_map: ``transformers`` device map (e.g. ``"auto"``), or None.
        trust_remote_code: Forwarded to ``from_pretrained``.
        tokenizer_name: Tokenizer id if different from ``model_name``.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str | None = None,
        device_map: str | None = None,
        trust_remote_code: bool = False,
        tokenizer_name: str | None = None,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dtype = dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.tokenizer_name = tokenizer_name or model_name
        self.load_in_4bit = load_in_4bit

    def load(self) -> ModelBundle:
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype) if self.dtype else None
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, trust_remote_code=self.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # QLoRA: load the base in NF4 4-bit. bitsandbytes places the weights via a
        # device_map, so pin to one GPU (LOCAL_RANK) and skip the .to() below.
        quant_kwargs: dict[str, object] = {}
        device_map = self.device_map
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
            )
            if device_map is None:
                device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=self.trust_remote_code,
            **quant_kwargs,
        )
        if not self.load_in_4bit and self.device_map is None and torch.cuda.is_available():
            # Under a distributed launch (torchrun), each process owns one GPU
            # (cuda:LOCAL_RANK); otherwise use the default device.
            local_rank = os.environ.get("LOCAL_RANK")
            device = f"cuda:{local_rank}" if local_rank is not None else "cuda"
            model = model.to(device)
        return model, tokenizer


@MODEL_PROVIDERS.register("cloud")
class CloudModelProvider(ModelProvider):
    """Placeholder for a remote/cloud-served model.

    Registered so configs can target a cloud backend; the concrete client is to
    be implemented when a cloud environment is adopted.
    """

    def __init__(self, endpoint: str, model_name: str, **options: Any) -> None:
        self.endpoint = endpoint
        self.model_name = model_name
        self.options = options

    def load(self) -> ModelBundle:
        raise NotImplementedError(
            "CloudModelProvider is a placeholder; implement the remote client "
            "before targeting a cloud backend."
        )
