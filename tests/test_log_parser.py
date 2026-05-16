"""
Tests for LogParser — covers all four log format strategies.
These tests are pure Python, no LLM or vector DB required.
"""

import pytest
from src.log_parser import LogParser, LogEntry


@pytest.fixture
def parser():
    return LogParser()


# --------------------------------------------------------------------------- #
#  pytest log format                                                           #
# --------------------------------------------------------------------------- #

PYTEST_LOG = """\
============================= test session starts ==============================
platform linux -- Python 3.11.4, pytest-7.4.3
collected 5 items

tests/test_api.py::test_health PASSED                                    [ 20%]
tests/test_api.py::test_create_user FAILED                               [ 40%]
tests/test_auth.py::test_login FAILED                                    [ 60%]

================================= FAILURES ====================================
_________________________ test_create_user _________________________

E       ConnectionRefusedError: [Errno 111] Connection refused
E       Could not connect to localhost:5432

tests/test_api.py:47: ConnectionRefusedError

========================= 2 failed, 1 passed in 3.21s =========================
"""

class TestPytestParser:
    def test_detects_pytest_format(self, parser):
        entry = parser.parse(PYTEST_LOG)
        assert entry.log_type == "pytest"

    def test_status_is_failed(self, parser):
        entry = parser.parse(PYTEST_LOG)
        assert entry.status == "failed"

    def test_extracts_failure_names(self, parser):
        entry = parser.parse(PYTEST_LOG)
        assert any("test_create_user" in f for f in entry.failures)
        assert any("test_login" in f for f in entry.failures)

    def test_extracts_error_messages(self, parser):
        entry = parser.parse(PYTEST_LOG)
        assert any("ConnectionRefusedError" in e or "Connection refused" in e
                   for e in entry.error_messages)

    def test_summary_contains_failed_count(self, parser):
        entry = parser.parse(PYTEST_LOG)
        assert "2" in entry.summary or "failed" in entry.summary.lower()

    def test_passed_log_has_passed_status(self, parser):
        passed_log = "===== test session starts =====\ncollected 1 item\ntest_foo.py::test_bar PASSED\n===== 1 passed in 0.5s ====="
        entry = parser.parse(passed_log)
        assert entry.status == "passed"


# --------------------------------------------------------------------------- #
#  Jest / npm log format                                                       #
# --------------------------------------------------------------------------- #

JEST_LOG = """\
 FAIL  src/components/UserCard.test.js
 PASS  src/utils/format.test.js

  ● UserCard > renders avatar

    TypeError: Cannot read properties of null

      at Object.<anonymous> (src/components/UserCard.test.js:19:4)

Test Suites: 1 failed, 1 passed, 2 total
Tests:       1 failed, 5 passed, 6 total
Time:        2.1s
npm ERR! Test failed.
"""

class TestJestParser:
    def test_detects_jest_format(self, parser):
        entry = parser.parse(JEST_LOG)
        assert entry.log_type == "jest"

    def test_status_is_failed(self, parser):
        entry = parser.parse(JEST_LOG)
        assert entry.status == "failed"

    def test_extracts_failed_suite(self, parser):
        entry = parser.parse(JEST_LOG)
        assert any("UserCard" in f for f in entry.failures)

    def test_extracts_error_type(self, parser):
        entry = parser.parse(JEST_LOG)
        assert any("TypeError" in e or "renders avatar" in e
                   for e in entry.error_messages)


# --------------------------------------------------------------------------- #
#  GitHub Actions log format                                                   #
# --------------------------------------------------------------------------- #

ACTIONS_LOG = """\
2024-01-01T10:00:00Z Run pip install -r requirements.txt
##[error]flask-sqlalchemy 2.5.1 requires Flask<3.0 but you have Flask 3.0.2
##[warning]Dependency conflicts detected
Process completed with exit code 1
##[error]The process failed with exit code 1
"""

class TestGithubActionsParser:
    def test_detects_actions_format(self, parser):
        entry = parser.parse(ACTIONS_LOG)
        assert entry.log_type == "github_actions"

    def test_status_is_failed(self, parser):
        entry = parser.parse(ACTIONS_LOG)
        assert entry.status == "failed"

    def test_extracts_error_annotations(self, parser):
        entry = parser.parse(ACTIONS_LOG)
        assert len(entry.error_messages) >= 1
        assert any("flask" in e.lower() or "exit code" in e.lower()
                   for e in entry.error_messages)

    def test_extracts_failed_exit(self, parser):
        entry = parser.parse(ACTIONS_LOG)
        assert any("1" in f for f in entry.failures)

    def test_timestamp_extracted(self, parser):
        entry = parser.parse(ACTIONS_LOG)
        assert entry.timestamp is not None
        assert "2024" in entry.timestamp


# --------------------------------------------------------------------------- #
#  Generic fallback format                                                     #
# --------------------------------------------------------------------------- #

GENERIC_LOG = """\
Starting application server...
Connecting to cache...
ERROR: Redis connection failed at redis://localhost:6379
FATAL: Cannot start worker pool without cache connection
"""

class TestGenericParser:
    def test_detects_generic_format(self, parser):
        entry = parser.parse(GENERIC_LOG)
        assert entry.log_type == "generic"

    def test_extracts_errors(self, parser):
        entry = parser.parse(GENERIC_LOG)
        assert len(entry.error_messages) >= 1
        assert any("Redis" in e or "cache" in e.lower() for e in entry.error_messages)

    def test_status_is_failed(self, parser):
        entry = parser.parse(GENERIC_LOG)
        assert entry.status == "failed"


# --------------------------------------------------------------------------- #
#  LogEntry.to_dict()                                                          #
# --------------------------------------------------------------------------- #

class TestLogEntryToDict:
    def test_returns_dict(self, parser):
        entry = parser.parse(PYTEST_LOG)
        d = entry.to_dict()
        assert isinstance(d, dict)

    def test_has_required_keys(self, parser):
        entry = parser.parse(PYTEST_LOG)
        d = entry.to_dict()
        for key in ("log_type", "status", "failures", "error_messages", "summary"):
            assert key in d

    def test_failures_is_list(self, parser):
        entry = parser.parse(PYTEST_LOG)
        assert isinstance(entry.to_dict()["failures"], list)


# --------------------------------------------------------------------------- #
#  Edge cases                                                                  #
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_empty_log(self, parser):
        entry = parser.parse("")
        assert entry.log_type == "generic"
        assert isinstance(entry.failures, list)
        assert isinstance(entry.error_messages, list)

    def test_very_long_log_truncated(self, parser):
        huge_log = "FAILED test_foo::bar\nE  SomeError: happened\n" * 500
        entry = parser.parse(huge_log)
        assert len(entry.raw_log) <= 4500   # raw_log is capped at 4000 chars

    def test_parse_returns_log_entry(self, parser):
        entry = parser.parse("some random text")
        assert isinstance(entry, LogEntry)
