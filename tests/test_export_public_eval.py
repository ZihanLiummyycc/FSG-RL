"""Publication-boundary tests for the locked evaluation exporter."""

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_public_eval.py"
SPEC = importlib.util.spec_from_file_location("export_public_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicEvalExportTests(unittest.TestCase):
    def test_private_fields_are_excluded(self):
        source = {
            "id": "omni_math_example", "text": "Question", "source": "omni",
            "difficulty": 3, "split": "locked_test", "gold_answer": "42",
            "metadata": {
                "do_not_train": True,
                "function_graph": {"nodes": [{"id": "main"}], "edges": []},
                "verification_graph": {"tests": ["secret"]},
                "verifier_annotation": {"reference_code": "secret"},
            },
        }
        result = MODULE.public_row(source)
        self.assertEqual(result["function_graph"]["nodes"][0]["id"], "main")
        self.assertNotIn("gold_answer", result)
        self.assertNotIn("verification_graph", str(result))
        self.assertNotIn("secret", str(result))

    def test_private_key_inside_public_graph_is_rejected(self):
        source = {
            "id": "omni_math_example", "text": "Question", "split": "locked_test",
            "metadata": {
                "do_not_train": True,
                "function_graph": {"nodes": [{"id": "main", "target_call": "secret"}]},
            },
        }
        with self.assertRaisesRegex(ValueError, "Private graph keys"):
            MODULE.public_row(source)


if __name__ == "__main__":
    unittest.main()
