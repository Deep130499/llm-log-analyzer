"""
Ingest past issues into the ChromaDB vector store.

Run once before using the analyzer to populate the RAG knowledge base:

    python scripts/ingest_issues.py

Options:
    --data-file   Path to JSON file (default: data/past_issues.json)
    --db-path     ChromaDB persist directory (default: ./chroma_db)
    --clear       Wipe the store before ingesting (re-seed from scratch)
"""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import track

# Make sure the project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


@click.command()
@click.option("--data-file", default="data/past_issues.json",
              show_default=True, help="Path to past_issues JSON file.")
@click.option("--db-path", default="./chroma_db",
              show_default=True, envvar="CHROMA_DB_PATH",
              help="ChromaDB persist directory.")
@click.option("--clear", is_flag=True, default=False,
              help="Clear the existing store before ingesting.")
def main(data_file: str, db_path: str, clear: bool):
    """Seed the RAG vector store with past CI/CD issues."""
    from src.vector_store import VectorStore  # imported here after sys.path tweak

    data_path = Path(data_file)
    if not data_path.exists():
        console.print(f"[red]Error:[/red] Data file not found: {data_path}")
        sys.exit(1)

    # ── Load JSON ────────────────────────────────────────────────────────────
    with data_path.open() as f:
        raw_issues = json.load(f)

    console.print(f"[cyan]→ Loaded {len(raw_issues)} issues from[/cyan] {data_path}")

    # ── Connect to store ─────────────────────────────────────────────────────
    console.print(f"[cyan]→ Connecting to ChromaDB at[/cyan] {db_path}")
    store = VectorStore(persist_dir=db_path)

    if clear:
        console.print("[yellow]→ Clearing existing collection…[/yellow]")
        store.clear()
        console.print("  Done.")

    existing = store.count()
    console.print(f"[dim]  Current document count: {existing}[/dim]")

    # ── Prepare batch ────────────────────────────────────────────────────────
    batch = []
    for issue in raw_issues:
        # Compose the text that will be embedded (richer = better retrieval)
        text_parts = [issue.get("title", "")]
        if issue.get("description"):
            text_parts.append(issue["description"])
        if issue.get("error_type"):
            text_parts.append(f"Error type: {issue['error_type']}")
        if issue.get("log_type"):
            text_parts.append(f"Log type: {issue['log_type']}")

        text = "\n".join(text_parts)

        metadata = {
            "title":      issue.get("title", ""),
            "resolution": issue.get("resolution", ""),
            "error_type": issue.get("error_type", ""),
            "log_type":   issue.get("log_type", ""),
            "severity":   issue.get("severity", ""),
            "tags":       issue.get("tags", []),  # VectorStore will stringify this
        }

        batch.append({
            "id":       issue.get("id"),
            "text":     text,
            "metadata": metadata,
        })

    # ── Ingest ───────────────────────────────────────────────────────────────
    console.print("[bold green]→ Computing embeddings and ingesting…[/bold green]")
    console.print("[dim]  (Downloads ~22 MB model on first run — cached afterwards)[/dim]")

    # Ingest individually so we can show progress
    ids_added = []
    for item in track(batch, description="  Ingesting…"):
        issue_id = store.add_issue(item["text"], item["metadata"], item.get("id"))
        ids_added.append(issue_id)

    # ── Summary ──────────────────────────────────────────────────────────────
    console.print()
    console.print(f"[bold green]✓ Done![/bold green] "
                  f"Ingested [bold]{len(ids_added)}[/bold] issues "
                  f"into [bold]{db_path}[/bold]")
    console.print(f"  Total documents in store: [bold]{store.count()}[/bold]")
    console.print()
    console.print("[dim]Run the analyzer now:[/dim]")
    console.print("  python -m src.analyzer -f logs/sample_pytest_failure.log")


if __name__ == "__main__":
    main()
