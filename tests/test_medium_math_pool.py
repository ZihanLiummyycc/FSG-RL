from __future__ import annotations

import unittest

from scripts.prepare_medium_math_pool import (
    assign_internal_splits,
    balanced_select,
    canonicalize_gsm8k,
    canonicalize_math,
    canonicalize_mathqa,
    extract_last_boxed_value,
    parse_mathqa_options,
)


class MediumMathPoolTests(unittest.TestCase):
    def test_gsm8k_requires_multistep_explicit_answer(self):
        row = {
            "question": "A three-step word problem.",
            "answer": (
                "First <<2+3=5>>5. Then <<5*4=20>>20. "
                "Finally <<20-2=18>>18. #### 18"
            ),
        }
        record = canonicalize_gsm8k(row)
        self.assertIsNotNone(record)
        self.assertEqual(record["gold_answer"], "18")
        self.assertEqual(record["difficulty"], 3)

    def test_nested_boxed_value(self):
        self.assertEqual(
            extract_last_boxed_value(r"Work. \boxed{\frac{3}{7}}"),
            r"\frac{3}{7}",
        )
        record = canonicalize_math(
            {
                "problem": "Compute a fraction.",
                "solution": r"Therefore \boxed{\frac{3}{7}}.",
                "level": "Level 2",
                "type": "Algebra",
            },
            config_name="algebra",
        )
        self.assertEqual(record["gold_answer"], r"\frac{3}{7}")

    def test_mathqa_numeric_option_and_program(self):
        options = "a ) 24 , b ) 120 , c ) 625 , d ) 720 , e ) 1024"
        self.assertEqual(parse_mathqa_options(options)["c"], "625")
        record = canonicalize_mathqa(
            {
                "Problem": "How many ways?",
                "Rationale": "Compute the product.",
                "options": options,
                "correct": "c",
                "linear_formula": "multiply(n0,n1)|power(#0,n2)|",
                "annotated_formula": "power(5,4)",
                "category": "general",
            }
        )
        self.assertIsNotNone(record)
        self.assertEqual(record["gold_answer"], "625")
        self.assertEqual(record["difficulty"], 2)

    def test_balanced_selection_and_splits_are_deterministic(self):
        rows = [
            {"id": f"p{index}", "source": "a" if index < 6 else "b", "group": index % 2}
            for index in range(10)
        ]
        selected = balanced_select(
            rows,
            6,
            seed=42,
            group_key=lambda row: (row["source"], row["group"]),
        )
        first = assign_internal_splits(
            selected,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=42,
        )
        second = assign_internal_splits(
            selected,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(len({row["id"] for row in first}), 6)


if __name__ == "__main__":
    unittest.main()
