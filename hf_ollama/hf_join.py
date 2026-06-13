import os
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    base_model = os.getenv(
        "PCB_EXPERT_BASE_MODEL",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    )
    adapter_path = os.getenv("PCB_EXPERT_ADAPTER_PATH")
    if not adapter_path:
        raise RuntimeError("PCB_EXPERT_ADAPTER_PATH must point to a local LoRA adapter")

    save_path = Path(os.getenv("PCB_MERGED_MODEL_DIR", "merged_deepseek"))
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main()
