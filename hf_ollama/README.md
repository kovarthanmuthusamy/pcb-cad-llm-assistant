# Exporting a local LoRA adapter

1. Set `PCB_EXPERT_BASE_MODEL` and `PCB_EXPERT_ADAPTER_PATH` as described in `.env.example`.
2. Run `python hf_ollama/hf_join.py` to merge the adapter into `PCB_MERGED_MODEL_DIR` (defaults to
   `merged_deepseek`).
3. Use the current `llama.cpp` Hugging Face conversion script to create a GGUF file. Follow the
   version-specific command in the upstream `llama.cpp` documentation rather than relying on a
   machine-specific path.
4. Register the resulting GGUF model with Ollama if local orchestration is required.

Model weights and adapters are intentionally excluded from this repository. Confirm the base
model, training-data, and adapter licences before redistribution.
