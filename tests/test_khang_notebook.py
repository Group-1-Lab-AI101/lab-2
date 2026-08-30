"""Presentation and execution checks for Khang's notebook."""

from __future__ import annotations

import json
import unittest

from src.data import PROJECT_ROOT


class KhangNotebookTests(unittest.TestCase):
    """Keep the Khang notebook executable and tied to the shared modules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = PROJECT_ROOT / "notebooks" / "khang_data_eda_preprocessing.ipynb"
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_notebook_covers_the_owned_pipeline_sections(self) -> None:
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        for section in (
            "Load and validate",
            "Target distribution",
            "Numerical and categorical EDA",
            "stratified train/test split",
            "Leakage-safe shared preprocessing",
            "Leakage and handoff audit",
            "Deployment caveat",
        ):
            self.assertIn(section, markdown)
        for shared_entry_point in (
            "load_bank_data",
            "make_stratified_split",
            "build_preprocessing_pipeline",
            "build_model_pipeline",
            "generate_eda_figures",
        ):
            self.assertIn(shared_entry_point, code)
        self.assertNotIn("train_test_split(", code)

    def test_notebook_is_executed_without_errors(self) -> None:
        code_cells = [
            cell
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        self.assertTrue(code_cells)
        self.assertTrue(
            all(cell.get("execution_count") is not None for cell in code_cells)
        )
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
