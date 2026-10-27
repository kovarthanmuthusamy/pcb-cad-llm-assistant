import io
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("flask")

from pcb_assistant.config import Settings  # noqa: E402
from pcb_assistant.rag.webui import app as webui  # noqa: E402


@pytest.fixture
def client_and_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    calls: list[tuple[str, ...]] = []

    def fake_run(module: str, *args: str) -> subprocess.CompletedProcess[str]:
        calls.append((module, *args))
        return subprocess.CompletedProcess([], 0, stdout="RESULT-FROM-CLI", stderr="")

    monkeypatch.setattr(webui, "_model_names", lambda: ["mistral:7b"])
    monkeypatch.setattr(webui, "_run_module", fake_run)
    app = webui.create_app(Settings(rag_home=tmp_path))
    return app.test_client(), calls, tmp_path


def test_index_page_renders(client_and_calls: Any) -> None:
    client, _, _ = client_and_calls
    response = client.get("/")
    assert response.status_code == 200
    assert b"mistral:7b" in response.data


def test_question_is_forwarded_to_the_document_cli(client_and_calls: Any) -> None:
    client, calls, _ = client_and_calls
    response = client.post(
        "/", data={"input": "get_result", "question": "What is TDR?", "model": "mistral:7b"}
    )
    assert b"RESULT-FROM-CLI" in response.data
    assert calls == [("pcb_assistant.rag.cli_pdf", "What is TDR?", "get_result", "mistral:7b")]


def test_uploaded_file_names_cannot_escape_the_upload_folder(client_and_calls: Any) -> None:
    client, calls, home = client_and_calls
    client.post(
        "/",
        data={
            "input": "store_data",
            "question": "q",
            "model": "m",
            "files": (io.BytesIO(b"pdf"), "../../evil.pdf"),
        },
        content_type="multipart/form-data",
    )
    assert (home / "data" / "evil.pdf").read_bytes() == b"pdf"
    assert not (home.parent / "evil.pdf").exists()
    assert calls[0][-1] == str(home / "data" / "evil.pdf")


def test_web_form_is_forwarded_to_the_web_cli(client_and_calls: Any) -> None:
    client, calls, _ = client_and_calls
    client.post(
        "/web",
        data={
            "input": "store_data",
            "query": "tdr",
            "question": "q",
            "search": "bing",
            "num_result": "3",
        },
    )
    assert calls == [
        (
            "pcb_assistant.rag.cli_web",
            "store_data",
            "tdr",
            "false",
            "q",
            "deepseek-r1:32b",
            "bing",
            "3",
        )
    ]


def test_main_binds_to_the_configured_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[dict[str, Any]] = []

    class FakeApp:
        def run(self, **kwargs: Any) -> None:
            runs.append(kwargs)

    settings = Settings(ui_host="0.0.0.0", ui_port=8080)
    monkeypatch.setattr(webui, "get_settings", lambda: settings)
    monkeypatch.setattr(webui, "create_app", lambda s: FakeApp())
    webui.main()
    assert runs == [{"host": "0.0.0.0", "port": 8080, "debug": False}]
