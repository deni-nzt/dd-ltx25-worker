# dd-ltx25-motion serverless worker — env + code BAKED IN (the proper way), weights on the
# RunPod network volume (large sequential reads only). Built by GitHub Actions on amd64.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PATH=/root/.local/bin:$PATH
RUN apt-get update -qq && apt-get install -y -qq git ffmpeg curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
RUN git clone --depth 1 --branch v1.2.0 https://github.com/Lightricks/LTX-2.git
WORKDIR /app/LTX-2
# uv manages its own python + the LTX lockfile env — identical to the proven pod worker.
RUN uv python install 3.11 && uv sync --frozen && uv pip install hf_transfer runpod && uv cache clean

COPY handler.py /app/handler.py
ENV LTX_MODELS_DIR=/runpod-volume/models LTX_QUANT=fp8-cast
CMD ["uv", "run", "python", "/app/handler.py"]
