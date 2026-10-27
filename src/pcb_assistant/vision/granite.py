"""Describe PCB images (circuit topologies, simulation plots) with Granite-Vision.

The model is IBM Granite-Vision-3.2-2B in 4-bit NF4, optionally with a local LoRA adapter
(``PCB_VISION_ADAPTER_PATH``; adapters are not distributed with this repository). Nothing heavy
is imported or loaded until :func:`describe_image` is called for the first time.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pcb_assistant.config import Settings, get_settings

logger = logging.getLogger(__name__)

TARGET_SIZE = (512, 512)
PROMPT = (
    "You are useful for describing the images related to PCB design domain "
    "eg.(circuit images (star or p2p topology),simulation results images), "
    "give me the detailed description of the image."
)


def to_pil_image(image: Any) -> Any:
    """Accept a PIL image, a file path or a NumPy array and return a PIL image."""

    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        return Image.open(image)
    import numpy as np

    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise TypeError("Unsupported image format. Provide a PIL image, NumPy array, or file path.")


def downscale(image: Any, target_size: tuple[int, int] = TARGET_SIZE) -> Any:
    """Convert to RGB and reduce the resolution (high-quality LANCZOS resampling)."""

    from PIL import Image

    return image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS).copy()


@lru_cache(maxsize=1)
def _load_model(settings: Settings) -> tuple[Any, Any, Any]:
    if settings.vision_cuda_device is not None:
        # Must be set before CUDA is initialised, i.e. before torch touches the GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.vision_cuda_device

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["vision_tower", "lm_head"],
    )
    model = AutoModelForVision2Seq.from_pretrained(
        settings.vision_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
    )
    processor = AutoProcessor.from_pretrained(settings.vision_model_id)

    if settings.vision_adapter_path:
        model.load_adapter(settings.vision_adapter_path)
    else:
        logger.warning(
            "PCB_VISION_ADAPTER_PATH is not set; using the base model without the adapter"
        )

    runnable = torch.compile(model) if settings.vision_compile else model
    runnable.eval()
    return model, runnable, processor


def describe_image(image: Any, settings: Settings | None = None) -> str:
    """Return ``" description = <text>"`` for a PCB-domain image."""

    settings = settings or get_settings()
    model, runnable, processor = _load_model(settings)
    picture = downscale(to_pil_image(image))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": picture},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=prompt_text, images=picture, return_tensors="pt").to(model.device)

    generated = runnable.generate(**inputs, max_new_tokens=settings.vision_max_new_tokens)
    new_tokens = [
        output[len(prompt_ids) :]
        for prompt_ids, output in zip(inputs.input_ids, generated, strict=True)
    ]
    texts = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return " description = " + " ".join(texts)
