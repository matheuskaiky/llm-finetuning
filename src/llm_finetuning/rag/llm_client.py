"""Local instruct-LLM client for the RAG pipeline (lazy transformers).

Wraps a chat model behind a small ``chat(messages) -> str`` interface so the
extractor, generator, critic and judge all share one swappable engine. Defaults
to greedy decoding (temperature 0) for reproducible extraction and judging.
"""

from __future__ import annotations

from typing import Any

_ROLE_LABEL = {"system": "Instrucoes", "user": "Usuario", "assistant": "Assistente"}


def _plain_prompt(messages: list[dict[str, str]]) -> str:
    """Render chat messages as a plain role-tagged prompt (base models with no
    chat template)."""
    lines = [f"{_ROLE_LABEL.get(m['role'], m['role'])}: {m['content']}" for m in messages]
    lines.append("Assistente:")
    return "\n\n".join(lines)


class LocalChatLLM:
    """A local HuggingFace instruct model exposed as a chat function."""

    def __init__(
        self,
        model_name: str = "models/Qwen3-8B",
        device: str = "cuda",
        device_map: str | dict | None = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model: Any = None
        self._tokenizer: Any = None

    @classmethod
    def from_config(cls, cfg: Any) -> LocalChatLLM:
        """Build from an ``LlmConfig`` (or any object with the same fields)."""
        return cls(
            model_name=cfg.model_name,
            device=cfg.device,
            device_map=getattr(cfg, "device_map", None),
            load_in_8bit=getattr(cfg, "load_in_8bit", False),
            load_in_4bit=getattr(cfg, "load_in_4bit", False),
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {"torch_dtype": "auto"}
        if self.load_in_8bit or self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        # device_map splits across GPUs (inference model-parallel, no NCCL); a
        # quantized load also requires a device_map. Otherwise pin to one device.
        if self.device_map or self.load_in_8bit or self.load_in_4bit:
            kwargs["device_map"] = self.device_map or "auto"
            # Forbid CPU/disk offload: cap each GPU and omit "cpu" so the model is
            # placed entirely on the GPUs or load fails loudly. A single offloaded
            # layer would make multi-GPU generation crawl (every token hits CPU).
            if not (self.load_in_8bit or self.load_in_4bit) and torch.cuda.is_available():
                n = torch.cuda.device_count()
                free = min(torch.cuda.mem_get_info(i)[0] for i in range(n))
                budget = int(free / 2**30 * 0.92)
                kwargs["max_memory"] = {i: f"{budget}GiB" for i in range(n)}

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if "device_map" not in kwargs and torch.cuda.is_available():
            self._model = self._model.to(self.device)
        self._model.eval()

    def chat(self, messages: list[dict[str, str]], max_new_tokens: int | None = None) -> str:
        """Run a chat completion and return the assistant text.

        Qwen3 "thinking" mode is disabled so outputs are direct (clean JSON for
        extraction, a bare score for judging).
        """
        import torch

        self._ensure_loaded()
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):
            try:
                prompt = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                # enable_thinking is Qwen-specific; other families reject it.
                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            # Base models have no chat template: build a plain role-tagged prompt.
            prompt = _plain_prompt(messages)
        inputs = tok(prompt, return_tensors="pt").to(self._model.device)
        do_sample = self.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tok.pad_token_id or tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()

    def complete(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Raw text completion of ``prompt`` (no chat template).

        Used to elicit answers from base and SFT models with the same instruction
        template, so before/after comparison uses an identical prompt format.
        """
        import torch

        self._ensure_loaded()
        tok = self._tokenizer
        inputs = tok(prompt, return_tensors="pt").to(self._model.device)
        do_sample = self.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tok.pad_token_id or tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()

    def response_perplexity(self, prompt: str, output: str) -> float:
        """Teacher-forced perplexity of ``output`` given ``prompt`` (response only).

        Masks the prompt tokens so the loss covers only the reference response;
        measures how well the model predicts the gold answer.
        """
        import math

        import torch

        self._ensure_loaded()
        tok = self._tokenizer
        prompt_ids = tok(prompt, add_special_tokens=True)["input_ids"]
        output_ids = tok(output, add_special_tokens=False)["input_ids"]
        if not output_ids:
            return float("nan")
        input_ids = torch.tensor([prompt_ids + output_ids], device=self._model.device)
        labels = torch.tensor(
            [[-100] * len(prompt_ids) + output_ids], device=self._model.device
        )
        with torch.no_grad():
            loss = self._model(input_ids, labels=labels).loss
        return float(math.exp(loss.item()))

    def unload(self) -> None:
        """Free the model/tokenizer and release GPU memory."""
        import gc

        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
