from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Hugging Face token — read from the environment (set HF_TOKEN in your shell or a .env file).
# Never hardcode credentials.
huggingface_token = os.getenv("HF_TOKEN")

# Login programmatically (only if a token is provided)
if huggingface_token:
    login(token=huggingface_token)


MODEL_NAME = os.getenv(
    "PCB_EXPERT_BASE_MODEL",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
)


def _load_expert():
    """Load the base model and local LoRA adapter only when the tool is invoked."""

    adapter_path = os.getenv("PCB_EXPERT_ADAPTER_PATH")
    if not adapter_path:
        raise RuntimeError(
            "PCB_EXPERT_ADAPTER_PATH is required; model adapters are not distributed in GitHub"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(model, adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

system_prompt = """You are an PCB design adviser sub-agent that provides helpful, concise,detailed answers to requested query. only provide the answer as paragraph, dont't include extra prompt or lines"""

def get_response(prompt, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        model, tokenizer, device = _load_expert()
    else:
        device = next(model.parameters()).device

    combined_prompt = f"PCB design expert: {prompt}\nAnswer:"

    inputs = tokenizer(combined_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
    inputs['input_ids'],
    max_length=512,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
    
    # Decode and clean
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the output
    output_text = output_text.replace(combined_prompt, "").strip()
    
    return output_text



