"""Checks for the public GRPO training-data export."""

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_training_data.py"
SPEC = importlib.util.spec_from_file_location("export_training_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrainingDataExportTests(unittest.TestCase):
    def test_training_verifier_is_retained_but_audit_code_removed(self):
        source = {
            "id": "one", "text": "Question", "gold_answer": "42",
            "metadata": {
                "function_graph": {"nodes": []},
                "verification_graph": {"nodes": [{"verification_spec": {"tests": []}}]},
                "verifier_annotation": {"reference_code": "not for release"},
                "verifier_execution_audit": {"message": "not for release"},
            },
        }
        result = MODULE.training_row(source)
        self.assertIn("verification_graph", result["metadata"])
        self.assertNotIn("verifier_annotation", result["metadata"])
        self.assertNotIn("verifier_execution_audit", result["metadata"])


if __name__ == "__main__":
    unittest.main()
