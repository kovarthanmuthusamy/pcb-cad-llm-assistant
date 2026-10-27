"""Local web UI: upload documents, index them and query the document and web stores.

Run with ``python -m pcb_assistant.rag.webui.app``. Every request runs the matching command
line module in a separate process (``cli_pdf`` / ``cli_web``), which keeps model and database
handles out of the web process. This is a local development tool and has no authentication.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from pcb_assistant.config import Settings, get_settings

PDF_ACTIONS = ["get_result", "delete_chroma", "delete_data", "store_data"]
WEB_ACTIONS = ["get_result", "store_data", "delete_chroma"]
SEARCH_ENGINES = ["duckDuckGo", "google", "brave", "bing", "yahoo", "startpage"]


def _model_names() -> list[str]:
    import ollama

    return [model["model"] for model in ollama.list()["models"]]


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args], capture_output=True, text=True, check=False
    )


def create_app(settings: Settings | None = None) -> Any:
    from flask import Flask, render_template, request
    from werkzeug.utils import secure_filename

    settings = settings or get_settings()
    upload_dir = settings.data_upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        models = _model_names()
        if request.method != "POST":
            return render_template(
                "index.html", output=None, files=[], models=models, inputs=PDF_ACTIONS
            )

        action = request.form.get("input", "get_result")
        question = request.form.get("question", "What is eCadstar")
        model = request.form.get("model", "mistral:7b")

        saved_files: list[str] = []
        for upload in request.files.getlist("files"):
            name = secure_filename(upload.filename or "")
            if name:
                path = upload_dir / name
                upload.save(path)
                saved_files.append(str(path))

        result = _run_module("pcb_assistant.rag.cli_pdf", question, action, model, *saved_files)
        return render_template(
            "index.html",
            output=result.stdout or result.stderr,
            files=saved_files,
            models=models,
            inputs=PDF_ACTIONS,
        )

    @app.route("/web", methods=["GET", "POST"])
    def web() -> str:
        models = _model_names()
        context = {"models": models, "inputs": WEB_ACTIONS, "searcher": SEARCH_ENGINES}
        if request.method != "POST":
            return render_template("web.html", output=None, **context)

        form = request.form
        result = _run_module(
            "pcb_assistant.rag.cli_web",
            form.get("input", "get_result"),
            form.get("query", "What is eCadstar"),
            form.get("isLink", "false"),
            form.get("question", "What is eCadstar"),
            form.get("model", "deepseek-r1:32b"),
            form.get("search", "duckDuckGo"),
            str(form.get("num_result", 2)),
        )
        return render_template("web.html", output=result.stdout or result.stderr, **context)

    return app


def main() -> None:
    settings = get_settings()
    # 127.0.0.1 by default; the Docker image sets PCB_UI_HOST=0.0.0.0 and Compose publishes
    # the port on the host's loopback interface only.
    create_app(settings).run(host=settings.ui_host, port=settings.ui_port, debug=settings.ui_debug)


if __name__ == "__main__":
    main()
