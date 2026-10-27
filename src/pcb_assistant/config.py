"""Runtime settings, read once from the environment (and a local ``.env`` file).

Every model name, path and timing that used to be hard-coded in the scripts lives here. The
defaults are the values the original scripts used, so a run on a new machine only needs
environment variables (see ``.env.example``).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _read(env: Mapping[str, str], name: str, default: T, cast: Callable[[str], T]) -> T:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid {cast.__name__} value") from exc


def _to_bool(raw: str) -> bool:
    value = raw.lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"expected one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}")


def _optional(env: Mapping[str, str], name: str) -> str | None:
    raw = env.get(name)
    return raw.strip() if raw and raw.strip() else None


@dataclass(frozen=True)
class Settings:
    """All tunable values of the assistant. Build one with :meth:`from_env`."""

    # Orchestrator back-ends
    ollama_model: str = "deepseek-r1:70b"
    openai_model: str = "gpt-4o"
    google_model: str = "gemini-1.5-pro"
    temperature: float = 0.7

    # Retrieval-augmented generation
    rag_home: Path = Path("rag_data")
    rag_model: str = "deepseek-r1:32b"
    rag_router_model: str = "deepseek-r1:32b"
    rag_num_results: int = 5
    rag_search_engine: str = "duckDuckGo"
    rag_cleanup_delay_s: float = 2.0
    embedding_model: str = "nomic-embed-text"
    user_agent: str = "MyCustomUserAgent/1.0"

    # PCB expert (base model + LoRA adapter)
    expert_base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    expert_adapter_path: str | None = None
    hf_token: str | None = None

    # Vision (image -> text)
    vision_model_id: str = "ibm-granite/granite-vision-3.2-2b"
    vision_adapter_path: str | None = None
    vision_max_new_tokens: int = 4000
    vision_cuda_device: str | None = None
    vision_compile: bool = True

    # Simulation (eCADSTAR on Windows)
    sim_min_similarity: float = 0.3
    ecadstar_p2p_sidf: str | None = None
    ecadstar_p2p_template: str | None = None
    ecadstar_star_sidf: str | None = None
    ecadstar_star_template: str | None = None
    ecadstar_output_dir: Path = Path("simulation_images")
    ecadstar_open_wait_s: float = 10.0
    ecadstar_run_wait_s: float = 10.0

    # Local web UI for the RAG corpus
    ui_debug: bool = False

    # --- derived locations -------------------------------------------------------------
    @property
    def chroma_pdf_dir(self) -> Path:
        return self.rag_home / "chroma"

    @property
    def chroma_web_dir(self) -> Path:
        return self.rag_home / "chroma_web"

    @property
    def chroma_default_dir(self) -> Path:
        return self.rag_home / "chroma_def"

    @property
    def data_pdf_dir(self) -> Path:
        return self.rag_home / "data_pdf"

    @property
    def data_default_dir(self) -> Path:
        return self.rag_home / "data_def"

    @property
    def data_upload_dir(self) -> Path:
        return self.rag_home / "data"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Read settings from ``environ`` (defaults to ``os.environ``)."""

        env = os.environ if environ is None else environ
        d = cls()  # defaults
        return cls(
            ollama_model=_read(env, "PCB_ORCHESTRATOR_MODEL", d.ollama_model, str),
            openai_model=_read(env, "PCB_OPENAI_MODEL", d.openai_model, str),
            google_model=_read(env, "PCB_GOOGLE_MODEL", d.google_model, str),
            temperature=_read(env, "PCB_TEMPERATURE", d.temperature, float),
            rag_home=_read(env, "PCB_RAG_HOME", d.rag_home, Path),
            rag_model=_read(env, "PCB_RAG_MODEL", d.rag_model, str),
            rag_router_model=_read(env, "PCB_RAG_ROUTER_MODEL", d.rag_router_model, str),
            rag_num_results=_read(env, "PCB_RAG_NUM_RESULTS", d.rag_num_results, int),
            rag_search_engine=_read(env, "PCB_RAG_SEARCH_ENGINE", d.rag_search_engine, str),
            rag_cleanup_delay_s=_read(env, "PCB_RAG_CLEANUP_DELAY_S", d.rag_cleanup_delay_s, float),
            embedding_model=_read(env, "PCB_EMBEDDING_MODEL", d.embedding_model, str),
            user_agent=_read(env, "PCB_USER_AGENT", d.user_agent, str),
            expert_base_model=_read(env, "PCB_EXPERT_BASE_MODEL", d.expert_base_model, str),
            expert_adapter_path=_optional(env, "PCB_EXPERT_ADAPTER_PATH"),
            hf_token=_optional(env, "HF_TOKEN"),
            vision_model_id=_read(env, "PCB_VISION_MODEL_ID", d.vision_model_id, str),
            vision_adapter_path=_optional(env, "PCB_VISION_ADAPTER_PATH"),
            vision_max_new_tokens=_read(
                env, "PCB_VISION_MAX_NEW_TOKENS", d.vision_max_new_tokens, int
            ),
            vision_cuda_device=_optional(env, "PCB_VISION_CUDA_DEVICE"),
            vision_compile=_read(env, "PCB_VISION_COMPILE", d.vision_compile, _to_bool),
            sim_min_similarity=_read(env, "PCB_SIM_MIN_SIMILARITY", d.sim_min_similarity, float),
            ecadstar_p2p_sidf=_optional(env, "ECADSTAR_P2P_SIDF"),
            ecadstar_p2p_template=_optional(env, "ECADSTAR_P2P_TEMPLATE"),
            ecadstar_star_sidf=_optional(env, "ECADSTAR_STAR_SIDF"),
            ecadstar_star_template=_optional(env, "ECADSTAR_STAR_TEMPLATE"),
            ecadstar_output_dir=_read(env, "ECADSTAR_OUTPUT_DIR", d.ecadstar_output_dir, Path),
            ecadstar_open_wait_s=_read(env, "ECADSTAR_OPEN_WAIT_S", d.ecadstar_open_wait_s, float),
            ecadstar_run_wait_s=_read(env, "ECADSTAR_RUN_WAIT_S", d.ecadstar_run_wait_s, float),
            ui_debug=_read(env, "PCB_UI_DEBUG", d.ui_debug, _to_bool),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Loads ``.env`` (if python-dotenv is installed) on first use."""

    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at runtime
        pass
    else:
        load_dotenv()
    return Settings.from_env()
