# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ─── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY src/       ./src/
COPY scripts/   ./scripts/
COPY data/      ./data/
COPY logs/      ./logs/

# Pre-download the sentence-transformers model during build so the
# container starts without needing network access at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ─── Environment defaults ─────────────────────────────────────────────────────
ENV LLM_PROVIDER=demo
ENV CHROMA_DB_PATH=/data/chroma_db
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Persist ChromaDB outside the container
VOLUME ["/data/chroma_db"]

# ─── Entrypoint ───────────────────────────────────────────────────────────────
# Default: show help. Override with docker run … --log-file /logs/ci.log
ENTRYPOINT ["python", "-m", "src.analyzer"]
CMD ["--help"]

# Usage examples:
#   docker build -t llm-log-analyzer .
#
#   # Analyze a local log file
#   docker run --rm \
#     -v "$(pwd)/logs:/logs:ro" \
#     -v "$(pwd)/chroma_db:/data/chroma_db" \
#     -e LLM_PROVIDER=demo \
#     llm-log-analyzer --log-file /logs/sample_pytest_failure.log
#
#   # With OpenAI
#   docker run --rm \
#     -v "$(pwd)/logs:/logs:ro" \
#     -v "$(pwd)/chroma_db:/data/chroma_db" \
#     -e LLM_PROVIDER=openai \
#     -e OPENAI_API_KEY=sk-... \
#     llm-log-analyzer --log-file /logs/ci.log
