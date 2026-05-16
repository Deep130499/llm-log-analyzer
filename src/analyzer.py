"""
Analyzer — main pipeline + CLI entry point.

Wires together the four core components:
  LogParser → VectorStore / RAGRetriever → LLMAnalyzer

CLI usage
---------
  # Analyze a log file (demo mode, no LLM required)
  python -m src.analyzer --log-file logs/sample_pytest_failure.log

  # Specify LLM provider
  python -m src.analyzer --log-file ci.log --provider ollama --model llama3.2
  python -m src.analyzer --log-file ci.log --provider openai

  # Read from stdin (useful in GitHub Actions)
  cat test_output.log | python -m src.analyzer --stdin

  # Output as Markdown (GitHub step summary compatible)
  python -m src.analyzer --log-file ci.log --format markdown

  # Pipe directly into $GITHUB_STEP_SUMMARY
  python -m src.analyzer --log-file test_output.log --format markdown >> $GITHUB_STEP_SUMMARY
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

from .log_parser import LogEntry, LogParser
from .llm_analyzer import LLMAnalyzer
from .rag_retriever import RAGRetriever, RetrievalResult
from .vector_store import VectorStore

console = Console()

# Default ChromaDB path (relative to where you run the CLI)
DEFAULT_DB_PATH = "./chroma_db"


# --------------------------------------------------------------------------- #
#  Core pipeline function (importable)                                        #
# --------------------------------------------------------------------------- #

def run_analysis(
    log_text: str,
    provider: str = "demo",
    model: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
    n_results: int = 3,
) -> dict:
    """
    Run the full analysis pipeline on raw log text.

    Returns a dict with keys:
        log_entry   : LogEntry (parsed log)
        retrieval   : RetrievalResult (similar past issues)
        analysis    : str (LLM-generated Markdown report)
    """
    # 1. Parse
    parser = LogParser()
    log_entry: LogEntry = parser.parse(log_text)

    # 2. Retrieve similar past issues
    store = VectorStore(persist_dir=db_path)
    retriever = RAGRetriever(store, n_results=n_results)
    retrieval: RetrievalResult = retriever.retrieve(log_entry)

    # 3. Analyze
    analyzer = LLMAnalyzer(provider=provider, model=model)
    analysis: str = analyzer.analyze(log_entry, retrieval)

    return {
        "log_entry": log_entry,
        "retrieval": retrieval,
        "analysis": analysis,
    }


# --------------------------------------------------------------------------- #
#  Output formatters                                                           #
# --------------------------------------------------------------------------- #

def _print_rich(log_entry: LogEntry, retrieval: RetrievalResult, analysis: str):
    """Full rich-formatted terminal output."""
    console.print()
    console.print(
        Panel.fit(
            "[bold white]🔍  LLM Log Analyzer — Analysis Report[/bold white]",
            border_style="bright_blue",
        )
    )

    # ── Parsed log summary ──────────────────────────────────────────────
    summary_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    summary_table.add_column("Key", style="dim")
    summary_table.add_column("Value", style="bold")

    status_color = "red" if log_entry.status == "failed" else "green"
    summary_table.add_row("Log Type", log_entry.log_type.upper())
    summary_table.add_row("Status",   f"[{status_color}]{log_entry.status.upper()}[/{status_color}]")
    summary_table.add_row("Summary",  log_entry.summary)
    summary_table.add_row("Provider", _llm_provider_label())

    console.print(Panel(summary_table, title="[bold]📋 Log Summary[/bold]", border_style="blue"))

    # ── Failures ────────────────────────────────────────────────────────
    if log_entry.failures:
        console.print("\n[bold red]❌  Failures[/bold red]")
        for f in log_entry.failures[:6]:
            console.print(f"  • [yellow]{f}[/yellow]")

    # ── RAG results ─────────────────────────────────────────────────────
    if retrieval.issues:
        console.print(f"\n[bold cyan]🗂  Similar Past Issues[/bold cyan] ({retrieval.count} found)")
        for i, issue in enumerate(retrieval.issues, 1):
            sim = int(issue.get("similarity", 0) * 100)
            doc = issue["document"][:100].replace("\n", " ")
            console.print(f"  {i}. [{sim}%] {doc}…")
            res = issue.get("metadata", {}).get("resolution", "")
            if res:
                console.print(f"       [dim]Fix: {res}[/dim]")
    else:
        console.print("\n[dim]🗂  No similar past issues in knowledge base (run scripts/ingest_issues.py)[/dim]")

    # ── LLM analysis ────────────────────────────────────────────────────
    console.print()
    console.print(Panel(Markdown(analysis), title="[bold green]🤖 AI Analysis[/bold green]", border_style="green"))
    console.print()


def _print_markdown(log_entry: LogEntry, retrieval: RetrievalResult, analysis: str):
    """GitHub-flavoured Markdown — suitable for $GITHUB_STEP_SUMMARY."""
    lines = [
        "# 🔍 LLM Log Analyzer — Analysis Report",
        "",
        "## 📋 Log Summary",
        "",
        f"| | |",
        f"|---|---|",
        f"| **Log Type** | `{log_entry.log_type}` |",
        f"| **Status** | `{log_entry.status.upper()}` |",
        f"| **Summary** | {log_entry.summary} |",
        "",
    ]

    if log_entry.failures:
        lines += ["## ❌ Failures", ""]
        for f in log_entry.failures[:6]:
            lines.append(f"- `{f}`")
        lines.append("")

    if retrieval.issues:
        lines += [f"## 🗂 Similar Past Issues ({retrieval.count} found)", ""]
        for i, issue in enumerate(retrieval.issues, 1):
            sim = int(issue.get("similarity", 0) * 100)
            doc = issue["document"][:120].replace("\n", " ")
            lines.append(f"{i}. **[{sim}% match]** {doc}")
            res = issue.get("metadata", {}).get("resolution", "")
            if res:
                lines.append(f"   > Fix: {res}")
        lines.append("")

    lines += ["## 🤖 AI Analysis", "", analysis, ""]

    print("\n".join(lines))


def _print_json(log_entry: LogEntry, retrieval: RetrievalResult, analysis: str):
    """Machine-readable JSON output."""
    output = {
        "log_entry": log_entry.to_dict(),
        "retrieval": {
            "query": retrieval.query,
            "count": retrieval.count,
            "issues": [
                {
                    "id": r["id"],
                    "similarity": r.get("similarity"),
                    "document": r["document"][:200],
                    "resolution": r.get("metadata", {}).get("resolution"),
                }
                for r in retrieval.issues
            ],
        },
        "analysis": analysis,
    }
    print(json.dumps(output, indent=2))


def _llm_provider_label() -> str:
    import os
    provider = os.getenv("LLM_PROVIDER", "demo")
    model    = os.getenv("LLM_MODEL", "")
    return f"{provider}" + (f" / {model}" if model else "")


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #

@click.command()
@click.option(
    "--log-file", "-f",
    type=click.Path(exists=True),
    default=None,
    help="Path to the CI/CD log file to analyze.",
)
@click.option(
    "--stdin", "use_stdin",
    is_flag=True,
    default=False,
    help="Read log from standard input.",
)
@click.option(
    "--provider", "-p",
    type=click.Choice(["ollama", "openai", "demo"], case_sensitive=False),
    default=None,
    envvar="LLM_PROVIDER",
    show_envvar=True,
    help="LLM provider to use (default: demo).",
)
@click.option(
    "--model", "-m",
    default=None,
    envvar="LLM_MODEL",
    show_envvar=True,
    help="Model name (e.g. llama3.2, gpt-3.5-turbo).",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["rich", "markdown", "json"], case_sensitive=False),
    default="rich",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--db-path",
    default=DEFAULT_DB_PATH,
    show_default=True,
    envvar="CHROMA_DB_PATH",
    help="Path to the ChromaDB persistent store.",
)
@click.option(
    "--n-results", "-n",
    default=3,
    show_default=True,
    help="Number of similar past issues to retrieve.",
)
def cli(
    log_file: Optional[str],
    use_stdin: bool,
    provider: Optional[str],
    model: Optional[str],
    output_format: str,
    db_path: str,
    n_results: int,
):
    """
    \b
    LLM-Powered CI/CD Log Analyzer
    ================================
    Parses CI/CD logs, retrieves similar past issues via RAG,
    and generates an AI-powered diagnosis.

    \b
    Examples:
      # Quick demo (no LLM needed)
      python -m src.analyzer -f logs/sample_pytest_failure.log

      # With local Ollama
      python -m src.analyzer -f ci.log --provider ollama --model llama3.2

      # Pipe to GitHub step summary
      python -m src.analyzer -f test_output.log --format markdown >> $GITHUB_STEP_SUMMARY
    """
    # -- Read log text -------------------------------------------------------
    if use_stdin:
        log_text = sys.stdin.read()
    elif log_file:
        log_text = Path(log_file).read_text(encoding="utf-8", errors="replace")
    else:
        console.print("[red]Error:[/red] Provide --log-file or --stdin.")
        sys.exit(1)

    if not log_text.strip():
        console.print("[yellow]Warning:[/yellow] Log input is empty.")
        sys.exit(0)

    # -- Run pipeline --------------------------------------------------------
    if output_format == "rich":
        with console.status("[bold green]Analyzing log…[/bold green]"):
            result = run_analysis(log_text, provider or "demo", model, db_path, n_results)
    else:
        result = run_analysis(log_text, provider or "demo", model, db_path, n_results)

    # -- Render output -------------------------------------------------------
    entry    = result["log_entry"]
    retrieval = result["retrieval"]
    analysis = result["analysis"]

    if output_format == "rich":
        _print_rich(entry, retrieval, analysis)
    elif output_format == "markdown":
        _print_markdown(entry, retrieval, analysis)
    elif output_format == "json":
        _print_json(entry, retrieval, analysis)

    # Exit non-zero if the log itself reported a failure
    if entry.status == "failed":
        sys.exit(1)


# Allow `python -m src.analyzer`
if __name__ == "__main__":
    cli()
