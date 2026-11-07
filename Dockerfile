FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch: the app only runs the small embedding model; the chat model
# is served by Ollama / vLLM in its own container. Saves ~2 GB of CUDA wheels.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .

# Run as an unprivileged user. Directories that hold generated data are
# created here so that named volumes inherit the right ownership.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p vectorstore clean_documents sources/cache sources/pdf /home/app/.cache/huggingface \
 && chown -R app:app /app /home/app/.cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
