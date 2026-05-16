"""
LLM Analyzer — sends parsed log data + RAG context to an LLM and
returns a structured diagnosis.

Supported providers
-------------------
  ollama  — local inference via Ollama (default, no API key needed)
  openai  — cloud inference via OpenAI API  (set OPENAI_API_KEY)
  demo    — template-based mock (no LLM required; great for CI)

The provider is selected via the LLM_PROVIDER environment variable
or explicitly in the constructor.

Usage:
    analyzer = LLMAnalyzer(provider="ollama", model="llama3.2")
    report   = analyzer.analyze(log_entry, retrieval_result)
    print(report)
"""

import os
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .log_parser import LogEntry
from .rag_retriever import RetrievalResult


# --------------------------------------------------------------------------- #
#  Prompt template                                                             #
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = (
    "You are an expert DevOps engineer and senior software developer. "
    "Your job is to diagnose CI/CD test failures, identify root causes, "
    "and provide concrete, actionable fix recommendations. "
    "Be specific — reference the exact test names, error messages, and files mentioned. "
    "Keep each section concise but complete."
)

_HUMAN_PROMPT = """\
## CI/CD Failure Report

**Log Type:** {log_type}
**Status:** {status}
**Summary:** {summary}

### Failed Tests / Jobs
{failures}

### Error Messages
{error_messages}

### Stack Traces
{stack_traces}

### Raw Log Excerpt
```
{raw_log}
```

---

## Similar Past Issues (Retrieved from Knowledge Base)
{similar_issues}

---

Please provide your analysis in the following format:

### Root Cause
[Identify the most likely cause — be specific about the error and its origin]

### Affected Components
[List affected files, modules, or services]

### Fix Recommendations
[Numbered list of concrete, actionable steps to resolve the issue]

### Preventive Measures
[How to prevent this class of failure in future pipelines]

### Priority
[Critical / High / Medium / Low — with a one-sentence justification]
"""


class LLMAnalyzer:
    """
    Orchestrates prompt construction, LLM invocation, and result parsing.

    Parameters
    ----------
    provider : str
        "ollama" | "openai" | "demo"  (env var LLM_PROVIDER overrides)
    model : str | None
        Model name; falls back to a sensible default per provider.
    """

    PROVIDER_DEFAULTS = {
        "ollama": "llama3.2",
        "openai": "gpt-3.5-turbo",
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "demo")).lower()
        self.model = model or os.getenv("LLM_MODEL") or self.PROVIDER_DEFAULTS.get(self.provider)
        self._chain = None

        if self.provider not in ("demo", "ollama", "openai"):
            raise ValueError(
                f"Unknown provider '{self.provider}'. Choose from: ollama, openai, demo."
            )

        if self.provider != "demo":
            self._build_chain()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze(self, log_entry: LogEntry, retrieval_result: RetrievalResult) -> str:
        """
        Run the full analysis pipeline and return a Markdown-formatted report.
        """
        if self.provider == "demo":
            return self._demo_analysis(log_entry, retrieval_result)

        payload = self._build_payload(log_entry, retrieval_result)
        return self._chain.invoke(payload)

    def stream_analyze(self, log_entry: LogEntry, retrieval_result: RetrievalResult):
        """
        Stream the LLM output token-by-token (generator).
        Falls back to a single yield in demo mode.
        """
        if self.provider == "demo":
            yield self._demo_analysis(log_entry, retrieval_result)
            return

        payload = self._build_payload(log_entry, retrieval_result)
        for chunk in self._chain.stream(payload):
            yield chunk

    # ------------------------------------------------------------------ #
    #  Chain construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
        )

        if self.provider == "ollama":
            # langchain-community Ollama wrapper
            from langchain_community.llms import Ollama  # noqa: PLC0415

            llm = Ollama(model=self.model, temperature=0.1)

        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI  # noqa: PLC0415

            llm = ChatOpenAI(
                model=self.model,
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        self._chain = prompt | llm | StrOutputParser()

    # ------------------------------------------------------------------ #
    #  Payload construction                                                #
    # ------------------------------------------------------------------ #

    def _build_payload(self, entry: LogEntry, retrieval: RetrievalResult) -> dict:
        def bullet_list(items, limit=6):
            if not items:
                return "  _None detected_"
            return "\n".join(f"  - {item}" for item in items[:limit])

        traces_text = ""
        for i, trace in enumerate(entry.stack_traces[:2], 1):
            traces_text += f"\n**Trace {i}:**\n```\n{trace[:500]}\n```"
        if not traces_text:
            traces_text = "  _No stack traces captured_"

        return {
            "log_type": entry.log_type,
            "status": entry.status.upper(),
            "summary": entry.summary,
            "failures": bullet_list(entry.failures),
            "error_messages": bullet_list(entry.error_messages),
            "stack_traces": traces_text,
            "raw_log": entry.raw_log[:2500],
            "similar_issues": retrieval.format_for_prompt(),
        }

    # ------------------------------------------------------------------ #
    #  Demo / mock analysis (no LLM required)                             #
    # ------------------------------------------------------------------ #

    def _demo_analysis(self, entry: LogEntry, retrieval: RetrievalResult) -> str:
        """
        Produce a template-driven analysis by pattern-matching the log
        entry.  Used in CI environments where no LLM is available and
        as a fallback during development.
        """
        failures_md = (
            "\n".join(f"  - `{f}`" for f in entry.failures[:5])
            or "  - _See error messages_"
        )
        errors_md = (
            "\n".join(f"  - {e}" for e in entry.error_messages[:4])
            or "  - _No explicit error strings captured_"
        )

        # Guess a likely root cause from the first error message
        root_cause = self._guess_root_cause(entry)

        # Build past-issue context block
        past_context = ""
        if retrieval.issues:
            past_context = "\n\n**Relevant past issues from the knowledge base:**\n"
            for issue in retrieval.issues[:3]:
                sim = int(issue.get("similarity", 0) * 100)
                doc = issue["document"][:120].replace("\n", " ")
                past_context += f"- [{sim}% match] {doc}\n"
                res = issue.get("metadata", {}).get("resolution")
                if res:
                    past_context += f"  ↳ _Past fix:_ {res}\n"

        return f"""\
### Root Cause
{root_cause}

### Affected Components
{failures_md}

### Fix Recommendations
1. Reproduce the failure locally:
   ```bash
   # pytest
   pytest {entry.failures[0] if entry.failures else 'tests/'} -vvs
   # jest
   npx jest --verbose {entry.failures[0] if entry.failures else ''}
   ```
2. Review the exact error messages:
{errors_md}
3. Check recent commits that touched the failing files (`git log --oneline -10`).
4. Verify all required environment variables and secrets are set in the workflow.
5. Confirm dependency versions are pinned and run `pip install -r requirements.txt` / `npm ci`.{past_context}

### Preventive Measures
- Add pre-commit hooks (`pre-commit install`) to catch lint/type errors before push.
- Pin all dependency versions to avoid unexpected breakage from upstream updates.
- Keep test coverage above 80% to surface regressions quickly.
- Enable branch protection rules: require all status checks to pass before merging.

### Priority
**High** — The CI pipeline is blocked. A broken `main` branch prevents team members
from merging work.  Investigate within the next working session.

---
> ⚠️  **Demo mode** — analysis generated from log patterns, not an LLM.
> Set `LLM_PROVIDER=ollama` (local) or `LLM_PROVIDER=openai` (cloud) for AI-powered analysis.
"""

    # ------------------------------------------------------------------ #
    #  Pattern-based root cause heuristics (demo mode only)               #
    # ------------------------------------------------------------------ #

    _HEURISTICS = [
        (r"ImportError|ModuleNotFoundError|No module named",
         "A Python **ImportError** suggests a missing or mis-installed package. "
         "Run `pip install -r requirements.txt` and verify the package is listed."),
        (r"AssertionError",
         "An **AssertionError** means a test assertion failed. "
         "The actual value did not match the expected value — "
         "check recent changes to the relevant function or fixture."),
        (r"ConnectionRefused|ECONNREFUSED|connection refused",
         "A **connection-refused error** indicates a service (database, API, cache) "
         "was not available when the test ran. "
         "Ensure required services are started in the CI workflow (e.g. `services:` block)."),
        (r"TimeoutError|timed out|ETIMEDOUT",
         "A **timeout** suggests a slow external dependency or an infinite loop. "
         "Check network calls, increase the timeout limit, or mock the external service in tests."),
        (r"PermissionError|EACCES|Permission denied",
         "A **permission error** likely means the CI runner lacks access to a file or socket. "
         "Review file permissions and ownership in the workflow."),
        (r"SyntaxError",
         "A **SyntaxError** means the code could not be parsed. "
         "Run `python -m py_compile <file>` locally to identify the line."),
        (r"TypeError",
         "A **TypeError** usually means a function received an argument of the wrong type, "
         "or a variable is `None` when an object was expected."),
        (r"KeyError|AttributeError",
         "A **KeyError / AttributeError** suggests a missing dictionary key or object attribute — "
         "often caused by a schema change or a renamed variable."),
        (r"npm ERR!|yarn error",
         "An **npm/yarn error** indicates a Node.js dependency or build issue. "
         "Try deleting `node_modules` and running `npm ci` for a clean install."),
        (r"exit code [^0]|non-zero exit",
         "A **non-zero exit code** means a shell command in the workflow failed. "
         "Check the step output above for the specific command that returned the error."),
    ]

    def _guess_root_cause(self, entry: LogEntry) -> str:
        import re as _re

        text = " ".join(entry.error_messages + entry.failures + [entry.raw_log[:500]])
        for pattern, explanation in self._HEURISTICS:
            if _re.search(pattern, text, _re.IGNORECASE):
                return explanation

        return (
            f"The `{entry.log_type}` log reports a **{entry.status}** status. "
            "No specific error pattern was matched automatically. "
            "Review the error messages and stack traces above to trace the failure to its source."
        )
