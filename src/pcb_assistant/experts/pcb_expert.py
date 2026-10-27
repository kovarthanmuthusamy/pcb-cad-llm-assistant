"""PCB / Signal-Integrity expert: DeepSeek-R1-Distill-Llama-8B with a local LoRA adapter.

The adapter is not distributed with this repository; point ``PCB_EXPERT_ADAPTER_PATH`` at it.
The model is loaded on first use and then kept in memory.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from pcb_assistant.config import Settings, get_settings

logger = logging.getLogger(__name__)

MAX_LENGTH = 512  # generation limit in tokens, prompt included


@lru_cache(maxsize=1)
def _load_expert(settings: Settings) -> tuple[Any, Any, str]:
    """Load the base model and the LoRA adapter once; later calls reuse them."""

    if not settings.expert_adapter_path:
        raise RuntimeError(
            "PCB_EXPERT_ADAPTER_PATH is required; model adapters are not distributed in GitHub"
        )

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if settings.hf_token:
        from huggingface_hub import login

        login(token=settings.hf_token)

    tokenizer = AutoTokenizer.from_pretrained(settings.expert_base_model)
    model = AutoModelForCausalLM.from_pretrained(settings.expert_base_model)
    model = PeftModel.from_pretrained(model, settings.expert_adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("PCB expert loaded on %s", device)
    return model, tokenizer, device


def get_response(
    prompt: str,
    model: Any = None,
    tokenizer: Any = None,
    settings: Settings | None = None,
) -> str:
    """Answer a PCB design question; loads the expert on first use if no model is given."""

    import torch

    if model is None or tokenizer is None:
        model, tokenizer, device = _load_expert(settings or get_settings())
    else:
        device = next(model.parameters()).device

    combined_prompt = f"PCB design expert: {prompt}\nAnswer:"
    inputs = tokenizer(combined_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=MAX_LENGTH,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_text: str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text.replace(combined_prompt, "").strip()
