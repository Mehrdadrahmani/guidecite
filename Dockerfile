# GuideCite API image (CPU). Works on linux/amd64 and linux/arm64.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    UV_SYSTEM_PYTHON=1

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# deps first for layer caching; CPU-only torch keeps the image ~2 GB smaller
COPY pyproject.toml requirements.lock README.md ./
RUN uv pip install --no-cache -r requirements.lock \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match

COPY configs ./configs
COPY prompts ./prompts
COPY src ./src
COPY scripts ./scripts
RUN uv pip install --no-cache --no-deps -e . && mkdir -p /app/data /app/.cache && chown -R app:app /app

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "medguide.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
