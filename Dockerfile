# syntax=docker/dockerfile:1
#
# Two targets:
#   test     the package with its dev tools; runs Ruff, mypy and pytest (no models, GPU or
#            eCADSTAR). Small and fast; CI builds and runs it on every push.
#   runtime  the assistant with the full model/RAG stack. CPU PyTorch by default; for a GPU build
#            pass --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 and run
#            with --gpus all.
#
#   docker build --target test -t pcb-assistant:test . && docker run --rm pcb-assistant:test
#   docker compose up -d ollama && docker compose run --rm assistant

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --create-home --uid 1000 app
WORKDIR /app


FROM base AS test
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[dev]"
COPY main.py ./
COPY tests ./tests
RUN chown -R app:app /app
USER app
CMD ["sh", "-c", "ruff check . && mypy && pytest"]


FROM base AS runtime
# libgl1 and libglib2.0-0 are needed by opencv-python, which the document loaders pull in.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.6,<3" "torchvision>=0.21,<1"

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Everything the assistant writes (vector stores, uploads, simulation images, model cache)
# goes to one volume. The web UI listens on all interfaces inside the container; Compose
# publishes it on the host's loopback interface only.
ENV PCB_RAG_HOME=/data/rag \
    ECADSTAR_OUTPUT_DIR=/data/simulation_images \
    HF_HOME=/data/huggingface \
    PCB_UI_HOST=0.0.0.0
RUN mkdir -p /data && chown app:app /data
VOLUME ["/data"]
USER app
EXPOSE 5000
ENTRYPOINT ["pcb-assistant"]
CMD ["--provider", "ollama"]
