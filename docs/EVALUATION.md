# Evaluation status

This repository is a research prototype. The evidence below distinguishes implemented components
from claims that still require a controlled benchmark.

## Implemented evidence

| Area | Evidence | Interpretation |
| --- | --- | --- |
| Orchestration | LangChain ReAct entry points for Ollama, GPT-4o, and Gemini with Pydantic tool schemas | Multiple model backends use the same tool contract |
| Retrieval | Local/PDF/web routing with Chroma-backed retrieval | Retrieval paths are implemented; no public relevance benchmark is claimed |
| Domain adaptation | Three-epoch LoRA runs for DeepSeek-R1-Distill-Llama-8B and Llama-3-8B | Adapter checkpoints and training logs exist locally; training loss is not task-quality evidence |
| CAD execution | 78 eCADSTAR simulation databases and 78 logs were generated during development | Demonstrates repeated tool execution; raw CAD artifacts are excluded from the public repository |
| Portable contracts | Deterministic tests for nested tool inputs and SI/FD/TDR parameter contracts | Runs in GitHub Actions without models, credentials, GPU, Windows, or eCADSTAR |

## Required next evaluation

A task-level benchmark should use a reviewed set of PCB/SI requests and report:

- Router accuracy across PCB expert, RAG, simulation, and vision actions
- Exact-match accuracy for analysis type, resistance, and line-length arguments
- Retrieval relevance and citation/groundedness checks
- Tool execution success and failure classification
- Expert-answer correctness against an engineering rubric and named baseline

The current repository does not claim end-to-end accuracy, production deployment, or superiority
over another assistant.
