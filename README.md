# 🔍 LLM-Powered CI/CD Log Analyzer with RAG

> Automatically analyze CI/CD test failures, summarize root causes, and retrieve
> similar past issues — powered by a local LLM (Ollama) and a vector database (ChromaDB).

[![CI Pipeline](https://github.com/YOUR_USERNAME/llm-log-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/llm-log-analyzer/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 What It Does

When a CI/CD pipeline fails, engineers spend time manually:
1. Reading dense log output
2. Identifying the root cause
3. Searching for similar issues they've solved before

This tool automates all three steps:

1. **Parses** logs from pytest, Jest/npm, GitHub Actions, and generic formats
2. **Retrieves** similar past issues from a vector database (RAG)
3. **Generates** a structured diagnosis — root cause, fix recommendations, and priority — using an LLM

The analysis is printed to the terminal **and** posted directly to the **GitHub Actions Step Summary** so it appears right on your PR.

---

## 🏗️ Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │           LLM Log Analyzer Pipeline          │
                         └─────────────────────────────────────────────┘

  CI/CD Log (raw text)
         │
         ▼
  ┌──────────────┐     Detect format      ┌─────────────────────────────┐
  │  Log Parser  │ ──────────────────────▶│  LogEntry (structured data) │
  │              │   pytest / jest /       │  - failures []              │
  │ log_parser.py│   github_actions /      │  - error_messages []        │
  └──────────────┘   generic              │  - stack_traces []          │
                                           │  - summary                  │
                                           └──────────────┬──────────────┘
                                                          │
                    ┌─────────────────────────────────────┘
                    │
         ┌──────────▼────────────┐
         │    RAG Retriever      │   Build semantic query
         │   rag_retriever.py    │ ──────────────────────▶  ChromaDB
         └──────────┬────────────┘                          (past issues)
                    │ top-k similar issues                    ▲
                    │                                         │
                    │                              ┌──────────┴────────────┐
                    │                              │   Vector Store        │
                    │                              │  vector_store.py      │
                    │                              │  sentence-transformers│
                    │                              │  all-MiniLM-L6-v2    │
                    │                              └───────────────────────┘
                    │
         ┌──────────▼────────────┐
         │    LLM Analyzer       │   Prompt = log data + similar issues
         │   llm_analyzer.py     │ ──────────────────────────────────────▶ LLM
         └──────────┬────────────┘                                 (Ollama / OpenAI)
                    │
                    ▼
         ┌──────────────────────┐
         │  Markdown Report     │
         │  - Root Cause        │ ──▶ Terminal (rich) / $GITHUB_STEP_SUMMARY / JSON
         │  - Fix Recommendations│
         │  - Priority          │
         └──────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/llm-log-analyzer.git
cd llm-log-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Seed the knowledge base

```bash
python scripts/ingest_issues.py
# → Ingests 15 sample past issues into ChromaDB
# → Downloads the ~22 MB all-MiniLM-L6-v2 embedding model (cached after first run)
```

### 3. Analyze a log

```bash
# Demo mode (no LLM needed — great for trying it out)
python -m src.analyzer --log-file logs/sample_pytest_failure.log

# Or run the full interactive demo
python scripts/demo.py
```

---

## 🤖 LLM Provider Setup

The tool supports three providers. Set `LLM_PROVIDER` in `.env` or as an environment variable.

### Option A — Ollama (local, free, private)

Best for: development, privacy-sensitive projects.

```bash
# 1. Install Ollama:  https://ollama.com
# 2. Pull a model
ollama pull llama3.2

# 3. Run the analyzer
LLM_PROVIDER=ollama python -m src.analyzer --log-file logs/sample_pytest_failure.log
```

### Option B — OpenAI (cloud)

Best for: highest quality analysis, CI/CD integration.

```bash
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai python -m src.analyzer --log-file logs/sample_pytest_failure.log
```

### Option C — Demo (no LLM)

Best for: CI environments, testing, and quick previews.

```bash
# Default — no setup required
python -m src.analyzer --log-file logs/sample_pytest_failure.log
```

---

## 📖 CLI Reference

```
Usage: python -m src.analyzer [OPTIONS]

Options:
  -f, --log-file PATH          Path to the CI/CD log file
      --stdin                  Read log from stdin
  -p, --provider [ollama|openai|demo]   LLM provider  [env: LLM_PROVIDER]
  -m, --model TEXT             Model name              [env: LLM_MODEL]
      --format [rich|markdown|json]     Output format  (default: rich)
      --db-path TEXT           ChromaDB path           [env: CHROMA_DB_PATH]
  -n, --n-results INT          Similar issues to retrieve (default: 3)
```

### Examples

```bash
# Analyze a pytest log in the terminal
python -m src.analyzer -f logs/sample_pytest_failure.log

# Analyze a Jest log with OpenAI and JSON output
python -m src.analyzer -f logs/sample_npm_failure.log --provider openai --format json

# Pipe from stdin (great for shell scripts)
cat /var/log/ci/run-123.log | python -m src.analyzer --stdin

# Output GitHub-flavoured Markdown to step summary
python -m src.analyzer -f ci.log --format markdown >> "$GITHUB_STEP_SUMMARY"

# Use with Docker
docker compose run --rm analyzer --log-file /logs/sample_pytest_failure.log
```

---

## 🔁 GitHub Actions Integration

### Automatic — CI pipeline

Push this repo to GitHub and the included workflow (`.github/workflows/ci.yml`)
will automatically:

1. Run the project's own tests on Python 3.10, 3.11, and 3.12
2. **If tests fail**, run the LLM Log Analyzer on the failure log
3. Post the AI-generated analysis to the **Actions Step Summary** (visible in the PR)

No secrets needed for demo mode. For OpenAI analysis, add `OPENAI_API_KEY` to your
repository secrets (`Settings → Secrets and variables → Actions`).

### Manual — analyze any log

Trigger the `Log Analyzer (Reusable)` workflow from the Actions tab:

1. Go to **Actions → Log Analyzer (Reusable) → Run workflow**
2. Paste your log content into the input field
3. Click **Run workflow**
4. View the analysis in the run's Step Summary

---

## 🐳 Docker

```bash
# Build
docker build -t llm-log-analyzer .

# Seed the knowledge base (run once)
docker compose run --rm ingest

# Analyze a log
docker compose run --rm analyzer --log-file /logs/sample_pytest_failure.log

# With Ollama (starts a local Ollama container)
docker compose --profile local-llm up -d ollama
docker compose exec ollama ollama pull llama3.2
LLM_PROVIDER=ollama docker compose run --rm analyzer -f /logs/sample_pytest_failure.log
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test module
pytest tests/test_log_parser.py -v
```

Tests use mocked embeddings — no model download needed.

---

## 📂 Project Structure

```
llm-log-analyzer/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Main CI pipeline + auto log analysis
│       └── log-analyzer.yml        # Reusable / manual analyzer workflow
│
├── src/
│   ├── __init__.py
│   ├── log_parser.py               # Multi-format log parser
│   ├── vector_store.py             # ChromaDB wrapper + sentence-transformers
│   ├── rag_retriever.py            # Semantic search + context formatting
│   ├── llm_analyzer.py             # Ollama / OpenAI / demo analysis
│   └── analyzer.py                 # Pipeline orchestrator + CLI entry point
│
├── data/
│   └── past_issues.json            # 15 seed issues for the RAG knowledge base
│
├── logs/
│   ├── sample_pytest_failure.log   # Sample pytest failure log
│   ├── sample_npm_failure.log      # Sample Jest/npm failure log
│   └── sample_actions_failure.log  # Sample GitHub Actions failure log
│
├── tests/
│   ├── test_log_parser.py          # Unit tests — parser strategies
│   ├── test_vector_store.py        # Unit tests — ChromaDB wrapper
│   └── test_pipeline.py            # Integration tests — full pipeline
│
├── scripts/
│   ├── ingest_issues.py            # Seed the vector DB from past_issues.json
│   └── demo.py                     # End-to-end demo runner
│
├── .env.example                    # Environment variable template
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| LLM (local)     | [Ollama](https://ollama.com) + llama3.2 |
| LLM (cloud)     | OpenAI GPT-3.5-turbo / GPT-4o       |
| Orchestration   | [LangChain](https://python.langchain.com) |
| Embeddings      | [sentence-transformers](https://sbert.net) `all-MiniLM-L6-v2` |
| Vector store    | [ChromaDB](https://www.trychroma.com) (persistent) |
| CLI             | [Click](https://click.palletsprojects.com) |
| Output          | [Rich](https://github.com/Textualize/rich) |
| CI/CD           | GitHub Actions                      |

---

## 🤝 Adding Your Own Past Issues

Edit `data/past_issues.json` and add entries following this schema:

```json
{
  "id": "issue-016",
  "title": "Brief one-line title of the issue",
  "description": "Longer description with error messages and context",
  "error_type": "ExceptionClassName",
  "log_type": "pytest",
  "resolution": "Concrete steps that fixed it",
  "tags": ["tag1", "tag2"],
  "severity": "high"
}
```

Then re-run:

```bash
python scripts/ingest_issues.py --clear
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
