"""Structural checks for Kiet's executed presentation notebook."""

from __future__ import annotations

import json
import unittest

from src.data import PROJECT_ROOT


class KietNotebookTests(unittest.TestCase):
    def test_notebook_is_executed_and_contains_no_error_outputs(self) -> None:
        notebook_path = (
            PROJECT_ROOT / "notebooks" / "kiet_pruning_visualization.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        error_outputs = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]

        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(code_cells), 8)
        self.assertTrue(all(cell["execution_count"] is not None for cell in code_cells))
        self.assertEqual(error_outputs, [])

    def test_notebook_covers_required_analysis_sections(self) -> None:
        notebook_path = (
            PROJECT_ROOT / "notebooks" / "kiet_pruning_visualization.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        content = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        for required_text in (
            "Thái Kiệt",
            "ccp_alpha",
            "Gini",
            "Entropy",
            "Confusion Matrix",
            "Kết luận chính",
        ):
            self.assertIn(required_text, content)

        for submission_only_text in (
            "quay video",
            "Run All Cells",
            "Kịch bản",
            "Checklist",
        ):
            self.assertNotIn(submission_only_text, content)


if __name__ == "__main__":
    unittest.main()
