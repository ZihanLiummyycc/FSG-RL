from __future__ import annotations

import unittest

from scripts.audit_teacher_judge_run import audit


class TeacherJudgeAuditTests(unittest.TestCase):
    def test_audit_detects_teacher_prompt_leak_and_score_variance(self):
        row = {
            "teacher_calls_used": 1,
            "teacher_group_judgment": {
                "rollout_judgments": [{"index": 0}, {"index": 1}]
            },
            "rollout_records": [
                {
                    "rollout": {
                        "prompt_text": "teacher-free prompt",
                        "completion_token_count": 10,
                    },
                    "reward": {"teacher_reward": 0.2},
                    "verification": {
                        "final_answer_score": 0.0,
                        "backward_score": 0.0,
                    },
                },
                {
                    "rollout": {
                        "prompt_text": "contains repair_feedback",
                        "completion_token_count": 2048,
                    },
                    "reward": {"teacher_reward": 0.9},
                    "verification": {
                        "final_answer_score": 1.0,
                        "backward_score": 1.0,
                    },
                },
            ],
            "grpo_update": {"approx_kl": 0.001},
        }
        result = audit([row], max_new_tokens=2048)
        self.assertEqual(result["teacher_group_judged_problems"], 1)
        self.assertEqual(result["student_prompts_containing_teacher_feedback"], 1)
        self.assertEqual(result["groups_with_teacher_score_variance"], 1)
        self.assertEqual(result["generations_at_token_cap"], 1)
        self.assertEqual(result["training_rollout_final_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
