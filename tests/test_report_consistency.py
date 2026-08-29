"""Ensure report metric blocks cannot silently diverge from artifacts."""

from __future__ import annotations

import unittest

from src.report_tables import assert_report_tables_current


class ReportConsistencyTests(unittest.TestCase):
    def test_generated_metric_tables_match_current_artifacts(self) -> None:
        assert_report_tables_current()


if __name__ == "__main__":
    unittest.main()
