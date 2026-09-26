from __future__ import annotations

import unittest
from unittest.mock import patch
import time

from fsg_rl.answer_equivalence import check_answer_equivalence
from fsg_rl.parsing import parse_tagged_function_spans
from fsg_rl.reward import RewardAssigner
from fsg_rl.schemas import ExecutionResult, FunctionGraph, FunctionNode, Problem
from fsg_rl.verifier import Verifier


class AnswerEquivalenceTests(unittest.TestCase):
    def assertEquivalent(self, left: str, right: str) -> None:
        result = check_answer_equivalence(left, right)
        self.assertTrue(result.equivalent, result)

    def assertNotEquivalent(self, left: str, right: str) -> None:
        result = check_answer_equivalence(left, right)
        self.assertFalse(result.equivalent, result)

    def test_root_expression(self):
        self.assertEquivalent(r"5(2 \sqrt{3}-3)", r"10 \sqrt{3}-15")

    def test_binomial_expression(self):
        self.assertEquivalent(r"63 \binom{64}{2}+1", "127009")

    def test_chained_inequality(self):
        self.assertEquivalent(
            r"{n \leq k \leq \lceil \tfrac32n \rceil}",
            r"n \leq k \leq \lceil \frac{3n}{2} \rceil",
        )

    def test_fraction_and_decimal(self):
        self.assertEquivalent(r"\frac{1}{2}", "0.5")

    def test_python_fraction_and_currency(self):
        self.assertEquivalent("Fraction(30, 1)", "$30")

    def test_numeric_answer_with_measurement_unit(self):
        self.assertEquivalent("82.0", r"82 \mathrm{~m}")
        self.assertEquivalent("176.0", r"176 \text{ cm}")
        self.assertEquivalent("10.0", "10 hours")
        self.assertNotEquivalent("82.0", r"83 \mathrm{~m}")

    def test_numeric_answer_with_thousands_separator(self):
        self.assertEquivalent("18000", "18,000")
        self.assertEquivalent("18000.5", "18,000.5")
        self.assertNotEquivalent("18000", "18,001")

    def test_numeric_answer_with_percent_suffix(self):
        self.assertEquivalent("22.22", "22.22 %")
        self.assertEquivalent("22.22", r"22.22\%")
        self.assertNotEquivalent("0.2222", "22.22%")
        self.assertNotEquivalent("22.22", "22.23%")

    def test_numeric_answer_with_degree_suffix(self):
        self.assertEquivalent("50.0", r"50^\circ")
        self.assertEquivalent("50", "50°")
        self.assertEquivalent("50", r"50^{\circ}")
        self.assertNotEquivalent("50", r"51^\circ")

    def test_numeric_complex_i_and_j_notation(self):
        self.assertEquivalent("(-4-2j)", "-4 - 2i")
        self.assertEquivalent("3+4j", "3+4i")
        self.assertEquivalent("j", "i")
        self.assertNotEquivalent("3+4j", "3-4i")
        self.assertNotEquivalent("-4-2j", "-4-3i")

    def test_explicit_sentence_final_scalar(self):
        self.assertEquivalent(
            "9",
            r"The smallest positive integer \(N\) satisfying the condition is \(9\).",
        )
        self.assertNotEquivalent("9", r"The statement discusses \(9\) possibilities.")

    def test_pi_and_power(self):
        self.assertEquivalent(r"2\pi+3^{2}", r"9+2\pi")

    def test_finite_set_is_unordered(self):
        self.assertEquivalent(r"\{1,2,\sqrt{9}\}", "{3,2,1}")

    def test_real_error_is_rejected(self):
        self.assertNotEquivalent("7184", "714")

    def test_unsupported_prose_fails_closed(self):
        self.assertNotEquivalent("the answer is seven", "7")

    def test_unsupported_relation_characters_fail_closed(self):
        result = check_answer_equivalence("x=@y", "x=y")
        self.assertFalse(result.equivalent)
        self.assertEqual(result.method, "unsupported")
        self.assertIn("ValueError", result.error or "")

    def test_symbolic_simplification_timeout_fails_closed(self):
        with patch(
            "fsg_rl.answer_equivalence._SYMBOLIC_TIMEOUT_SECONDS",
            0.05,
        ), patch("sympy.simplify", side_effect=lambda _value: time.sleep(1)):
            started = time.monotonic()
            result = check_answer_equivalence("x+x+1", "2*x+1")
        self.assertFalse(result.equivalent)
        self.assertIn("TimeoutError", result.error or "")
        self.assertLess(time.monotonic() - started, 0.5)

    def test_equivalent_final_answer_receives_final_reward(self):
        graph = FunctionGraph(
            problem_id="p1",
            nodes=[
                FunctionNode(
                    id="main",
                    name="answer",
                    question="Answer the problem.",
                    signature="main() -> str",
                    expected_output_type="final_answer",
                    verification_spec={},
                )
            ],
            edges=[],
        )
        problem = Problem(id="p1", text="test", gold_answer=r"10 \sqrt{3}-15")
        spans = parse_tagged_function_spans(
            r"<a_main>Therefore \boxed{5(2 \sqrt{3}-3)}.</a_main>",
            graph,
        )
        execution = ExecutionResult()
        verification = Verifier({}).verify(problem, graph, spans, execution)
        reward = RewardAssigner(
            {
                "reward": {
                    "lambda_final": 1.0,
                    "lambda_node": 0.0,
                    "lambda_edge": 0.0,
                    "lambda_backward": 0.0,
                    "lambda_consensus": 0.0,
                    "lambda_efficiency": 0.0,
                    "lambda_repair": 0.0,
                }
            }
        ).assign(problem, graph, verification, execution, spans)
        self.assertEqual(verification.final_answer_score, 1.0)
        self.assertEqual(reward.final_reward, 1.0)
        self.assertEqual(reward.total_reward, 1.0)

    def test_unverifiable_prose_node_does_not_get_default_full_credit(self):
        graph = FunctionGraph(
            problem_id="p1",
            nodes=[
                FunctionNode(
                    id="claim",
                    name="claim",
                    question="Prove a claim.",
                    signature="claim() -> str",
                    expected_output_type="proof_claim",
                    verification_spec={},
                ),
                FunctionNode(
                    id="main",
                    name="answer",
                    question="Answer.",
                    signature="main() -> int",
                    expected_output_type="final_answer",
                    verification_spec={},
                ),
            ],
            edges=[],
        )
        spans = parse_tagged_function_spans(
            r"<a_claim>plausible prose</a_claim><a_main>\boxed{2}</a_main>",
            graph,
        )
        verification = Verifier({}).verify(
            Problem(id="p1", text="1+1", gold_answer="2"),
            graph,
            spans,
            ExecutionResult(),
        )
        self.assertNotIn("claim", verification.node_scores)
        self.assertNotIn("main", verification.node_scores)


if __name__ == "__main__":
    unittest.main()
