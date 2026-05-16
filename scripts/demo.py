"""
Demo script — runs the full pipeline on all three sample logs
and prints the results with rich formatting.

Usage:
    python scripts/demo.py
    python scripts/demo.py --provider ollama --model llama3.2
    python scripts/demo.py --provider openai
"""

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.rule import Rule

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

SAMPLE_LOGS = [
    ("pytest",          "logs/sample_pytest_failure.log"),
    ("npm / jest",      "logs/sample_npm_failure.log"),
    ("GitHub Actions",  "logs/sample_actions_failure.log"),
]


@click.command()
@click.option("--provider", default="demo",
              type=click.Choice(["demo", "ollama", "openai"], case_sensitive=False),
              envvar="LLM_PROVIDER", show_envvar=True)
@click.option("--model", default=None, envvar="LLM_MODEL", show_envvar=True)
@click.option("--db-path", default="./chroma_db", envvar="CHROMA_DB_PATH", show_envvar=True)
@click.option("--skip-ingest", is_flag=True, default=False,
              help="Skip re-ingesting sample issues (faster if already done).")
def main(provider, model, db_path, skip_ingest):
    """Run a full end-to-end demo of the LLM Log Analyzer."""

    console.print()
    console.print(Rule("[bold cyan]🔍 LLM-Powered Log Analyzer — Demo[/bold cyan]"))
    console.print(f"  Provider : [bold]{provider}[/bold]" + (f" / {model}" if model else ""))
    console.print(f"  DB path  : {db_path}")
    console.print()

    # ── Step 1: Ingest past issues ───────────────────────────────────────────
    if not skip_ingest:
        console.print(Rule("[bold yellow]Step 1 — Seeding RAG Knowledge Base[/bold yellow]"))
        from src.vector_store import VectorStore
        import json

        store = VectorStore(persist_dir=db_path)
        data_file = Path("data/past_issues.json")

        if not data_file.exists():
            console.print(f"[red]Error:[/red] {data_file} not found. Run from project root.")
            sys.exit(1)

        with data_file.open() as f:
            issues = json.load(f)

        console.print(f"  Ingesting {len(issues)} past issues…")
        for issue in issues:
            text = f"{issue.get('title', '')}\n{issue.get('description', '')}"
            meta = {k: issue[k] for k in ("resolution", "error_type", "severity", "tags")
                    if k in issue}
            store.add_issue(text, meta, issue.get("id"))

        console.print(f"  [green]✓ Store contains {store.count()} documents[/green]")
        console.print()

    # ── Step 2: Analyze each sample log ─────────────────────────────────────
    from src.analyzer import run_analysis
    from src.analyzer import _print_rich  # reuse the rich formatter

    for i, (label, log_path) in enumerate(SAMPLE_LOGS, 1):
        lp = Path(log_path)
        if not lp.exists():
            console.print(f"[yellow]Skipping {log_path} — file not found[/yellow]")
            continue

        console.print(
            Rule(f"[bold]Step {i + 1} — Analyzing {label} log[/bold]")
        )
        console.print(f"  File: [dim]{log_path}[/dim]")
        console.print()

        log_text = lp.read_text(encoding="utf-8")

        t0 = time.perf_counter()
        with console.status(f"  Running pipeline…"):
            result = run_analysis(log_text, provider=provider, model=model, db_path=db_path)
        elapsed = time.perf_counter() - t0

        _print_rich(result["log_entry"], result["retrieval"], result["analysis"])
        console.print(f"  [dim]Analysis completed in {elapsed:.2f}s[/dim]")
        console.print()

    # ── Done ─────────────────────────────────────────────────────────────────
    console.print(Rule("[bold green]Demo Complete[/bold green]"))
    console.print()
    console.print("  Next steps:")
    console.print("  • Analyze your own logs:")
    console.print("    [cyan]python -m src.analyzer -f /path/to/your/ci.log[/cyan]")
    if provider == "demo":
        console.print()
        console.print("  • Enable AI-powered analysis:")
        console.print("    [cyan]LLM_PROVIDER=ollama python -m src.analyzer -f ci.log[/cyan]")
        console.print("    (requires Ollama running locally: https://ollama.com)")
    console.print()


if __name__ == "__main__":
    main()
