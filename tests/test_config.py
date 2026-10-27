from pathlib import Path

import pytest

from pcb_assistant.config import Settings


def test_defaults_are_the_original_hard_coded_values() -> None:
    settings = Settings.from_env({})
    assert settings.ollama_model == "deepseek-r1:70b"
    assert settings.openai_model == "gpt-4o"
    assert settings.temperature == 0.7
    assert settings.rag_model == "deepseek-r1:32b"
    assert settings.rag_num_results == 5
    assert settings.vision_max_new_tokens == 4000
    assert settings.sim_min_similarity == 0.3
    assert settings.ecadstar_open_wait_s == 10.0
    assert settings.ui_debug is False


def test_environment_overrides_defaults() -> None:
    settings = Settings.from_env(
        {
            "PCB_ORCHESTRATOR_MODEL": "deepseek-r1:32b",
            "PCB_TEMPERATURE": "0.2",
            "PCB_RAG_NUM_RESULTS": "3",
            "PCB_VISION_COMPILE": "false",
            "PCB_UI_DEBUG": "1",
            "ECADSTAR_P2P_SIDF": "C:/sim/p2p.sidf",
        }
    )
    assert settings.ollama_model == "deepseek-r1:32b"
    assert settings.temperature == 0.2
    assert settings.rag_num_results == 3
    assert settings.vision_compile is False
    assert settings.ui_debug is True
    assert settings.ecadstar_p2p_sidf == "C:/sim/p2p.sidf"


def test_blank_values_fall_back_to_defaults() -> None:
    settings = Settings.from_env({"PCB_TEMPERATURE": "  ", "PCB_EXPERT_ADAPTER_PATH": ""})
    assert settings.temperature == 0.7
    assert settings.expert_adapter_path is None


@pytest.mark.parametrize(
    ("name", "value"),
    [("PCB_TEMPERATURE", "warm"), ("PCB_RAG_NUM_RESULTS", "many"), ("PCB_UI_DEBUG", "maybe")],
)
def test_invalid_values_name_the_variable(name: str, value: str) -> None:
    with pytest.raises(ValueError, match=name):
        Settings.from_env({name: value})


def test_rag_locations_live_under_one_home() -> None:
    settings = Settings.from_env({"PCB_RAG_HOME": "state"})
    assert settings.rag_home == Path("state")
    assert settings.chroma_pdf_dir == Path("state/chroma")
    assert settings.chroma_web_dir == Path("state/chroma_web")
    assert settings.data_pdf_dir == Path("state/data_pdf")
    assert settings.data_upload_dir == Path("state/data")


def test_settings_are_hashable_for_caching() -> None:
    assert hash(Settings.from_env({})) == hash(Settings.from_env({}))
