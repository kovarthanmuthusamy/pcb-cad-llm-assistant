<div align="center">

# A Multi-Agent LLM Assistant for PCB Design CAD (eCADSTAR)

*An orchestrated, multimodal, tool-using chatbot that helps beginners learn and operate a
Signal-Integrity EDA workflow — answering domain questions, running live simulations, and
interpreting circuit & result images.*

</div>

[![CI](https://github.com/kovarthanmuthusamy/pcb-cad-llm-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/kovarthanmuthusamy/pcb-cad-llm-assistant/actions/workflows/ci.yml)

---

## Abstract

Electronic Design Automation (EDA) tools such as **eCADSTAR** are powerful but steep to learn: a
newcomer must simultaneously understand Signal-Integrity (SI) theory, the CAD UI, and how to read
simulation outputs. This project develops a **multi-agent Large Language Model (LLM) assistant** that
lowers that barrier. A central **orchestrator LLM** decomposes a user's request and routes it to a
set of specialized tools: a **domain-fine-tuned PCB expert**, a **Retrieval-Augmented Generation
(RAG)** agent over internal documents and the live web, a **simulation agent** that drives the
eCADSTAR Electrical Editor and returns results, and a **vision agent** that turns circuit and
simulation **images into text** so the (text-only) orchestrator can reason over them. The system is
**multimodal** (accepts text + images), **model-agnostic** (the orchestrator can be a local DeepSeek-R1
model via Ollama, OpenAI GPT-4o, or Google Gemini), and grounded in the SI sub-domain of EDA through
LoRA fine-tuning.

> **Context.** Group research project. Built with vision-language and reasoning models available in
> 2023–2024 that were strong at both agentic tool-use and image understanding.

### Contribution and ownership

Kovarthanam Muthusamy designed the LangChain/ReAct orchestration, implemented the RAG sub-agent,
connected the natural-language parameter extractor to eCADSTAR automation, and integrated the
project components into the assistant workflow. The vision component was developed by a project
colleague and integrated into this system; it is not presented as an individually authored model.
The team coordinated through Git; colleagues built a React user interface intended for use alongside
the eCADSTAR PCB editor, which this backend connects to. That interface and the Docker setup used for
reproducible environments are not part of this public showcase.

This public repository is a reviewed research showcase. Private corpora, model adapters,
checkpoints, Chroma databases, and raw eCADSTAR project/result files are deliberately excluded.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [System Architecture](#2-system-architecture)
3. [Query Lifecycle](#3-query-lifecycle)
4. [Components](#4-components)
   - [4.1 Orchestrator](#41-orchestrator-model-agnostic)
   - [4.2 PCB Expert (fine-tuned)](#42-pcb-expert-fine-tuned-sub-agent)
   - [4.3 RAG Agent](#43-rag-agent-documents--web)
   - [4.4 Simulation Agent](#44-simulation-agent-ecadstar)
   - [4.5 Image Descriptor (vision)](#45-image-descriptor-vision-agent)
5. [Repository Structure](#5-repository-structure)
6. [Setup & Usage](#6-setup--usage)
7. [Security Notes](#7-security-notes)
8. [Limitations & Future Work](#8-limitations--future-work)
9. [Verification & Evidence](#9-verification--evidence)

---

## 1. Motivation

A beginner using a Signal-Integrity CAD tool repeatedly hits three walls:

1. **Knowledge** — "What is a TDR analysis? Why does my net ring?"
2. **Operation** — "How do I actually run this simulation with these parameters?"
3. **Interpretation** — "What does this result waveform / circuit screenshot mean?"

A single LLM cannot reliably do all three: it lacks tool access, up-to-date/document-grounded facts,
and (for a text model) the ability to see images. This project addresses each wall with a dedicated
agent, coordinated by an orchestrator — a classic **planner + specialized-tools** architecture.

---

## 2. System Architecture

```mermaid
flowchart TB
    U["👤 User<br/>(text + up to 3 images)"] --> IMG["🖼️ Image Descriptor<br/>(Granite-Vision VLM)"]
    IMG -- "image → text" --> ORC
    U -- "text query" --> ORC

    ORC{{"🧠 Orchestrator LLM<br/>DeepSeek-R1 / GPT-4o / Gemini<br/>(LangChain ReAct + memory)"}}

    ORC -- "decompose & route<br/>(JSON action)" --> T1["📘 PCB Expert<br/>fine-tuned DeepSeek-R1-8B + LoRA"]
    ORC --> T2["🔎 RAG Agent<br/>docs + web (Chroma)"]
    ORC --> T3["⚙️ Simulation Agent<br/>eCADSTAR Electrical Editor"]

    T3 -- "result image" --> IMG2["🖼️ Image Descriptor"]
    IMG2 -- "result → text" --> ORC
    T1 -- "observation" --> ORC
    T2 -- "observation" --> ORC

    ORC -- "final answer" --> U

    style ORC fill:#1b5e20,color:#fff
    style IMG fill:#33415c,color:#fff
    style IMG2 fill:#33415c,color:#fff
```

The orchestrator follows a **ReAct** loop (Reason → Act → Observe), emitting a structured JSON
`{action, action_input}` per step (LangChain `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION`), and
keeps short-term context via `ConversationBufferMemory`. Because every tool returns **text**, even
inherently visual results (simulation plots) are first passed through the vision agent — making the
whole pipeline reason in a single textual space.

---

## 3. Query Lifecycle

```mermaid
sequenceDiagram
    actor User
    participant V as Image Descriptor (VLM)
    participant O as Orchestrator LLM
    participant E as PCB Expert
    participant R as RAG Agent
    participant S as Simulation Agent

    User->>V: query + image(s)
    V-->>O: textual image description(s)
    User->>O: text query
    Note over O: Reason: which tool?
    O->>S: "SI, R=100Ω, TL=100mm"
    S->>S: drive eCADSTAR, capture result image
    S->>V: result image
    V-->>S: textual description
    S-->>O: observation (result as text)
    O->>E: domain follow-up (SI theory)
    E-->>O: grounded explanation
    O-->>User: final synthesized answer
```

---

## 4. Components

### 4.1 Orchestrator (model-agnostic)

Three interchangeable back-ends share the **same tool set and prompt**:

| Entry point | Orchestrator model | Provider |
| ----------- | ------------------ | -------- |
| `main.py` | `deepseek-r1:70b` (or `:32b`) | **Local** via Ollama |
| `openai_main.py` | `gpt-4o` | OpenAI |
| `google_main.py` | `gemini-1.5-pro` | Google |

Tools are registered as LangChain `StructuredTool`s with Pydantic input schemas
(`langchain_Tools_v2.py`). A system instruction constrains the agent to **one action at a time**,
and post-processing strips the model's `<think>` traces and trailing JSON before display.

### 4.2 PCB Expert (fine-tuned sub-agent)

`PCB/PCB_expert.py` — a **DeepSeek-R1-Distill-Llama-8B** base model with a **LoRA adapter** fine-tuned
on domain question–answer pairs in the **Signal-Integrity area of EDA**. It answers PCB design,
manufacturing, and noise questions as concise paragraphs, and is the *primary* tool for any
PCB-related query. (A Llama-3-8B + LoRA variant is also included.)

### 4.3 RAG Agent (documents + web)

`RAGv2/` — a **router-based** retrieval agent. A lightweight LLM "chooser" (`chooser.py`) inspects
the available local corpora and the query, then selects a source:

```mermaid
flowchart LR
    Q["query"] --> C{"Chooser LLM<br/>pick source"}
    C -- "1" --> D["Local default<br/>Chroma store"]
    C -- "2" --> P["PDF documents<br/>(embed → Chroma)"]
    C -- "3" --> W["Live web search<br/>DuckDuckGo / Google CSE / Brave / Bing"]
    D & P & W --> A["Answer<br/>(retrieved-context LLM)"]
    style C fill:#1b5e20,color:#fff
```

Retrieved passages are embedded and stored in a **Chroma** vector database, then a generator model
(`deepseek-r1:32b`) answers from the retrieved context. Multiple search back-ends are supported
(`w_get_search.py`). This is the fallback when the fine-tuned expert lacks coverage and provides
**document- and web-grounded** facts.

### 4.4 Simulation Agent (eCADSTAR)

Turns natural language into an actual CAD simulation:

```mermaid
flowchart LR
    NL["'Run SI with 100Ω<br/>and 100mm line'"] --> SB["S-BERT semantic match<br/>(simtool_search.py)"]
    SB --> PARAMS["type ∈ {SI, FD, TDR}<br/>R_value, TL_value"]
    PARAMS --> CAD["eCADSTAR Electrical Editor<br/>GUI automation (pyautogui/win32)<br/>P2P topology"]
    CAD --> IMG["result image"]
    style SB fill:#33415c,color:#fff
    style CAD fill:#7c2d12,color:#fff
```

- **Parameter extraction** (`simulation_tool/simtool_search.py`): a Sentence-BERT model
  (`all-MiniLM-L6-v2`) maps free-text to the analysis type — **SI** (Signal Integrity),
  **FD** (Frequency Domain), **TDR** (Time-Domain Reflectometry) — and regex/embeddings pull the
  resistor (Ω) and transmission-line length (mm) values.
- **Execution** (`simulation_tool/sim_tool.py`): patches the SIDF/circuit file with those values and
  drives the eCADSTAR Electrical Editor via GUI automation on the default **P2P** topology, capturing
  the result image.
- The image is then described by the vision agent and returned to the orchestrator as text.

![Default P2P topology driven by the simulation agent: driver IC3, series resistor R2, 100 mm transmission line TL1, receiver IC4](docs/images/p2p-topology.png)

*Default P2P topology. The agent sets the series resistor (R2) and line length (TL1) from the user's request.*

### 4.5 Image Descriptor (vision agent)

`image_descriptor_tool/` — a **fine-tuned IBM Granite-Vision-3.2-2B** vision-language model
(4-bit NF4 quantized, LoRA adapter), prompted to describe PCB-domain images: circuit topologies
(star / P2P) and simulation result plots. It serves **two** roles: (1) converting user-attached
images into text at input, and (2) converting simulation **output** images into text. Alternative
VLM back-ends were explored and are included for comparison: **LLaVA-1.6-Mistral**,
**Qwen2.5-VL-7B**, and **SmolVLM-500M**.

---

## 5. Repository Structure

```text
pcb-cad-llm-assistant/
├── main.py                     orchestrator — local DeepSeek-R1 (Ollama)
├── openai_main.py              orchestrator — GPT-4o
├── google_main.py              orchestrator — Gemini 1.5 Pro
├── langchain_Tools_v2.py       tool definitions (Simulation, PCB Expert, RAG)
├── langchain_Tools.py          earlier tool definitions, kept for reference
├── template.py                 ReAct prompt template
├── setup_imports.py            path wiring
├── src/pcb_assistant/
│   └── contracts.py            portable, model-free tool-input contracts
├── tests/
│   └── test_contracts.py       contract tests run in CI
├── PCB/
│   └── PCB_expert.py           fine-tuned SI/EDA expert (DeepSeek-R1-8B + LoRA)
├── RAGv2/
│   ├── final.py                RAG entry (source routing)
│   ├── chooser.py              LLM source selector
│   ├── w_RAG_web.py            web retrieval
│   ├── f_RAG_pdf.py            PDF retrieval
│   └── w_get_search.py         search back-ends (DuckDuckGo/Google/Brave/…)
├── simulation_tool/
│   ├── simtool_search.py       S-BERT parameter extraction
│   └── sim_tool.py             eCADSTAR GUI automation
├── image_descriptor_tool/
│   ├── granite_vision32_inference.py   primary VLM (Granite-Vision)
│   ├── qwen25_inference.py / llava_1_6_mistral_inference.py / smolvlm.py
│   ├── llava_*_augmentation.py         data augmentation for VLM fine-tuning
│   └── *_Train.ipynb                   VLM fine-tuning notebooks
├── hf_ollama/
│   └── hf_join.py              merge a LoRA adapter for Ollama export
├── docs/
│   ├── EVALUATION.md           implemented evidence and benchmark plan
│   └── images/                 eCADSTAR topology and result examples
├── .github/workflows/ci.yml    Ruff + pytest on pushes and pull requests to main
├── pyproject.toml              package, Ruff and pytest configuration
├── requirements.txt
├── SECURITY.md
└── .env.example                template for required credentials
```

> Model weights/adapters, vector databases, private RAG corpora, and raw simulation databases/logs
> are **not** distributed (see `.gitignore`). Selected eCADSTAR topology and result images remain in
> `docs/images/` as documentation examples; they are not a substitute for a reproducible end-to-end benchmark.

---

## 6. Setup & Usage

**1. Environment**

```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
```

For the portable contract tests only—no model, GPU, credentials, Windows, or eCADSTAR required:

```bash
pip install -e ".[dev]"
ruff check src tests
pytest
```

`requirements.txt` lists portable top-level dependencies. GPU-specific PyTorch builds should be
selected for the target CUDA environment using the official PyTorch installation guidance.

**2. Credentials** — copy the template and fill in your own keys:

```bash
cp .env.example .env       # then edit .env
```

**3. Local models** (for the default orchestrator) — pull via [Ollama](https://ollama.com):

```bash
ollama pull deepseek-r1:70b     # primary
ollama pull deepseek-r1:32b     # lighter alternative / RAG generator
```

**4. Run** the interactive assistant:

```bash
python main.py            # local DeepSeek-R1 orchestrator
# or: python openai_main.py   (GPT-4o)
# or: python google_main.py   (Gemini)
```

At the prompt, type a query and optionally provide up to **3 image paths**. Simulation output images
are saved under `simulation_tool/simulation_images/`.

> The simulation agent automates the **eCADSTAR Electrical Editor** through the Windows GUI
> (`pyautogui`, `pygetwindow`, `win32gui`) and therefore requires a Windows host with eCADSTAR
> installed and the bundled circuit files present. The knowledge/RAG/vision agents run independently
> of the CAD tool.

---

## 7. Security Notes

- **No secrets in source.** All API keys (OpenAI, Google, Hugging Face) are read from environment
  variables / a local `.env` file, which is gitignored. Use `.env.example` as the template.
- If you forked an earlier version that contained hardcoded keys, **rotate those keys** — anything
  committed to git history must be considered compromised.
- Raw CAD projects, simulation databases/logs, local document corpora, and model weights are
  excluded until ownership and redistribution rights are confirmed.

---

## 8. Limitations & Future Work

- **Platform coupling.** The simulation agent depends on GUI automation of a specific Windows CAD
  install; a headless/scripting API would make it robust and portable.
- **Two-hop vision.** Visual results are flattened to text before reasoning; a natively multimodal
  orchestrator would remove the descriptor round-trip and preserve detail.
- **Single topology.** Simulation currently targets the default P2P topology; broadening to arbitrary
  circuits and more analysis types is a natural extension.
- **Modern agent framework.** Built on LangChain's classic agent; migrating to a graph-based
  controller (e.g. LangGraph) would give clearer multi-step control and error recovery.

---

## 9. Verification & Evidence

| Capability | Public implementation evidence | Current validation boundary |
| --- | --- | --- |
| Tool orchestration | ReAct entry points for Ollama, GPT-4o, and Gemini with Pydantic schemas | Provider calls require the user's credentials and model access |
| Retrieval | Local/PDF/web routing and Chroma-backed indexing/query code | No public retrieval-relevance benchmark is claimed |
| Domain adaptation | DeepSeek-R1-Distill-Llama-8B and Llama-3-8B LoRA integration | Three-epoch training runs exist locally; loss alone is not task-quality evidence |
| CAD automation | Sentence-BERT analysis selection, parameter extraction, and Windows GUI adapter | 78 simulation databases and 78 logs were generated locally but are excluded from GitHub |
| Portable contracts | Nested tool-input and SI/FD/TDR parameter tests under `tests/` | Enforced with Ruff and pytest in GitHub Actions |

See [docs/EVALUATION.md](docs/EVALUATION.md) for the exact implemented evidence, missing task-level
metrics, and the benchmark required before making end-to-end accuracy claims.

### Selected eCADSTAR outputs

| Signal-integrity waveform | TDR response | Frequency-domain response |
| --- | --- | --- |
| ![Signal-integrity waveform](docs/images/si-waveform.png) | ![TDR response](docs/images/tdr-response.png) | ![Frequency-domain response](docs/images/frequency-domain-response.png) |

---

<div align="center">
<sub>Multi-agent LLM assistant for Signal-Integrity EDA — research/educational prototype.</sub>
</div>
