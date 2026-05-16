"""
Log Parser — Detects and parses CI/CD logs from multiple formats:
  - pytest        (Python test runner)
  - jest/npm      (JavaScript test runner)
  - GitHub Actions (workflow runner output)
  - generic       (fallback for any unknown format)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LogEntry:
    """Structured representation of a parsed CI/CD log."""
    log_type: str                       # "pytest" | "jest" | "github_actions" | "generic"
    timestamp: Optional[str]
    status: str                         # "failed" | "passed" | "error" | "unknown"
    failures: List[str] = field(default_factory=list)       # test/job names that failed
    error_messages: List[str] = field(default_factory=list) # extracted error strings
    stack_traces: List[str] = field(default_factory=list)   # extracted stack traces
    raw_log: str = ""                   # truncated raw text for LLM context
    summary: str = ""                   # one-line human-readable summary

    def to_dict(self) -> dict:
        return {
            "log_type": self.log_type,
            "timestamp": self.timestamp,
            "status": self.status,
            "failures": self.failures,
            "error_messages": self.error_messages,
            "stack_traces": self.stack_traces,
            "summary": self.summary,
        }


class LogParser:
    """
    Detects the format of a CI/CD log and routes it to the
    appropriate parsing strategy.

    Usage:
        parser = LogParser()
        entry  = parser.parse(open("test_output.log").read())
        print(entry.summary)
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def parse(self, log_content: str) -> LogEntry:
        """Parse raw log text and return a structured LogEntry."""
        log_type = self._detect_log_type(log_content)

        parsers = {
            "pytest":          self._parse_pytest,
            "jest":            self._parse_jest,
            "github_actions":  self._parse_github_actions,
            "generic":         self._parse_generic,
        }

        return parsers.get(log_type, self._parse_generic)(log_content)

    # ------------------------------------------------------------------ #
    #  Format detection                                                    #
    # ------------------------------------------------------------------ #

    def _detect_log_type(self, log: str) -> str:
        if re.search(r"test session starts|pytest|PASSED|FAILED.*::", log):
            return "pytest"
        if re.search(r"npm (ERR!|run)|jest|Test Suites:|PASS |FAIL ", log):
            return "jest"
        if re.search(r"##\[error\]|##\[warning\]|::error::|Run .+\n", log):
            return "github_actions"
        return "generic"

    # ------------------------------------------------------------------ #
    #  pytest parser                                                       #
    # ------------------------------------------------------------------ #

    def _parse_pytest(self, log: str) -> LogEntry:
        # e.g. "FAILED tests/test_auth.py::test_login - AssertionError"
        failures = re.findall(r"FAILED\s+([\w/\\.\-:]+)", log)

        # Exception lines after "E " prefix used by pytest
        errors = re.findall(r"^E\s+(.+)", log, re.MULTILINE)
        errors = [e.strip() for e in errors if e.strip()][:8]

        traces = self._extract_python_tracebacks(log)

        # Result line: "3 failed, 9 passed in 4.23s"
        result_match = re.search(
            r"(\d+ failed.*?(?:passed|error).*?in [\d.]+s)", log
        )
        summary = result_match.group(1) if result_match else f"{len(failures)} test(s) failed"

        status = "failed" if failures or re.search(r"\d+ (failed|error)", log) else "passed"

        return LogEntry(
            log_type="pytest",
            timestamp=self._extract_timestamp(log),
            status=status,
            failures=failures,
            error_messages=errors,
            stack_traces=traces[:3],
            raw_log=log[:4000],
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    #  Jest / npm parser                                                   #
    # ------------------------------------------------------------------ #

    def _parse_jest(self, log: str) -> LogEntry:
        # "FAIL src/components/Button.test.js"
        failed_suites = re.findall(r"^FAIL\s+(.+)$", log, re.MULTILINE)
        # "✕ should render correctly (45 ms)"
        failed_tests = re.findall(r"[✕✗×]\s+(.+?)(?:\s+\(\d+\s*ms\))?$", log, re.MULTILINE)
        failures = failed_suites + failed_tests

        # "● Button > should render correctly"
        errors = re.findall(r"●\s+(.+)", log)[:8]

        # Jest summary: "Tests: 3 failed, 12 passed, 15 total"
        summary_match = re.search(r"Tests:\s+([\d\w ,]+total)", log)
        summary = summary_match.group(1) if summary_match else f"{len(failures)} test(s) failed"

        status = "failed" if failures or re.search(r"\d+ failed", log) else "passed"

        return LogEntry(
            log_type="jest",
            timestamp=self._extract_timestamp(log),
            status=status,
            failures=failures,
            error_messages=errors,
            stack_traces=self._extract_js_stacks(log)[:3],
            raw_log=log[:4000],
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    #  GitHub Actions parser                                               #
    # ------------------------------------------------------------------ #

    def _parse_github_actions(self, log: str) -> LogEntry:
        # Inline annotations
        errors = re.findall(r"(?:##\[error\]|::error[^:]*::)(.+)", log)
        warnings = re.findall(r"(?:##\[warning\]|::warning[^:]*::)(.+)", log)

        # Non-zero exit codes
        exit_codes = re.findall(r"Process completed with exit code (\d+)", log)
        failed_exits = [c for c in exit_codes if c != "0"]

        failures = [f"Exit code {c}" for c in failed_exits]
        failures += re.findall(r"##\[error\](.+)", log)[:5]

        status = "failed" if errors or failed_exits else "passed"
        summary = (
            f"{len(errors)} error(s), {len(warnings)} warning(s) in Actions workflow"
            if errors or warnings
            else "Workflow completed"
        )

        return LogEntry(
            log_type="github_actions",
            timestamp=self._extract_timestamp(log),
            status=status,
            failures=failures,
            error_messages=(errors + warnings)[:8],
            stack_traces=self._extract_python_tracebacks(log)[:2],
            raw_log=log[:4000],
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    #  Generic / fallback parser                                           #
    # ------------------------------------------------------------------ #

    def _parse_generic(self, log: str) -> LogEntry:
        errors = re.findall(
            r"(?:ERROR|FATAL|FAILED|CRITICAL|Exception|Error):\s*(.+)",
            log,
            re.IGNORECASE,
        )[:8]

        status = "failed" if errors else "unknown"
        summary = f"{len(errors)} error(s) detected" if errors else "No errors detected"

        return LogEntry(
            log_type="generic",
            timestamp=self._extract_timestamp(log),
            status=status,
            failures=[],
            error_messages=errors,
            stack_traces=self._extract_python_tracebacks(log)[:2],
            raw_log=log[:4000],
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                      #
    # ------------------------------------------------------------------ #

    def _extract_python_tracebacks(self, log: str) -> List[str]:
        """Extract Python-style Traceback blocks."""
        pattern = r"Traceback \(most recent call last\):.*?(?=\n\n|\Z)"
        matches = re.findall(pattern, log, re.DOTALL)
        return [m[:600] for m in matches]

    def _extract_js_stacks(self, log: str) -> List[str]:
        """Extract JS-style 'at ...' stack frames grouped by blank lines."""
        traces, current = [], []
        for line in log.splitlines():
            if re.match(r"\s+at\s+", line):
                current.append(line.strip())
            else:
                if current:
                    traces.append("\n".join(current))
                    current = []
        return [t[:600] for t in traces]

    def _extract_timestamp(self, log: str) -> Optional[str]:
        patterns = [
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?",
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
            r"\d{2}:\d{2}:\d{2}",
        ]
        for p in patterns:
            m = re.search(p, log)
            if m:
                return m.group(0)
        return None
