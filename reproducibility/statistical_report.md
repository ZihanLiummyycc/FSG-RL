# Locked Eval400 three-way evaluation

Records: 400

## Aggregate results

| Metric | stage2 | main_grpo | feedback_grpo | teacher_judge |
|---|---:|---:|---:|---:|
| tags_exact | 99.5% | 99.8% | 99.2% | 99.2% |
| code_blocks_complete | 99.5% | 99.8% | 99.2% | 99.2% |
| signature_exact | 99.5% | 99.8% | 99.2% | 99.2% |
| python_executable | 86.0% | 90.2% | 90.8% | 91.0% |
| all_hidden_tests_pass | 61.8% | 65.8% | 67.0% | 66.5% |
| backward_target_correct | 41.5% | 63.7% | 63.7% | 63.2% |
| final_answer_correct | 43.2% | 67.5% | 69.0% | 68.2% |
| full_success | 32.2% | 52.2% | 53.5% | 52.8% |

## Paired comparisons

### stage2_vs_main_grpo

| Metric | Difference | Improved | Regressed | McNemar p |
|---|---:|---:|---:|---:|
| tags_exact | +0.3% | 2 | 1 | 1.0000 |
| code_blocks_complete | +0.3% | 2 | 1 | 1.0000 |
| signature_exact | +0.3% | 2 | 1 | 1.0000 |
| python_executable | +4.2% | 25 | 8 | 0.0046 |
| all_hidden_tests_pass | +4.0% | 31 | 15 | 0.0259 |
| backward_target_correct | +22.2% | 106 | 17 | 0.0000 |
| final_answer_correct | +24.3% | 115 | 18 | 0.0000 (Holm 0.0000) |
| full_success | +20.0% | 95 | 15 | 0.0000 |

### stage2_vs_feedback_grpo

| Metric | Difference | Improved | Regressed | McNemar p |
|---|---:|---:|---:|---:|
| tags_exact | -0.2% | 1 | 2 | 1.0000 |
| code_blocks_complete | -0.2% | 1 | 2 | 1.0000 |
| signature_exact | -0.2% | 1 | 2 | 1.0000 |
| python_executable | +4.7% | 27 | 8 | 0.0019 |
| all_hidden_tests_pass | +5.2% | 33 | 12 | 0.0025 |
| backward_target_correct | +22.2% | 107 | 18 | 0.0000 |
| final_answer_correct | +25.7% | 117 | 14 | 0.0000 (Holm 0.0000) |
| full_success | +21.3% | 98 | 13 | 0.0000 |

### stage2_vs_teacher_judge

| Metric | Difference | Improved | Regressed | McNemar p |
|---|---:|---:|---:|---:|
| tags_exact | -0.2% | 2 | 3 | 1.0000 |
| code_blocks_complete | -0.2% | 2 | 3 | 1.0000 |
| signature_exact | -0.2% | 2 | 3 | 1.0000 |
| python_executable | +5.0% | 26 | 6 | 0.0005 |
| all_hidden_tests_pass | +4.7% | 29 | 10 | 0.0034 |
| backward_target_correct | +21.7% | 106 | 19 | 0.0000 |
| final_answer_correct | +25.0% | 117 | 17 | 0.0000 (Holm 0.0000) |
| full_success | +20.5% | 94 | 12 | 0.0000 |

### main_grpo_vs_feedback_grpo

| Metric | Difference | Improved | Regressed | McNemar p |
|---|---:|---:|---:|---:|
| tags_exact | -0.5% | 0 | 2 | 0.5000 |
| code_blocks_complete | -0.5% | 0 | 2 | 0.5000 |
| signature_exact | -0.5% | 0 | 2 | 0.5000 |
| python_executable | +0.5% | 5 | 3 | 0.7266 |
| all_hidden_tests_pass | +1.3% | 7 | 2 | 0.1797 |
| backward_target_correct | +0.0% | 7 | 7 | 1.0000 |
| final_answer_correct | +1.5% | 11 | 5 | 0.2101 |
| full_success | +1.3% | 8 | 3 | 0.2266 |

### main_grpo_vs_teacher_judge

| Metric | Difference | Improved | Regressed | McNemar p |
|---|---:|---:|---:|---:|
| tags_exact | -0.5% | 0 | 2 | 0.5000 |
| code_blocks_complete | -0.5% | 0 | 2 | 0.5000 |
| signature_exact | -0.5% | 0 | 2 | 0.5000 |
| python_executable | +0.8% | 3 | 0 | 0.2500 |
| all_hidden_tests_pass | +0.8% | 5 | 2 | 0.4531 |
| backward_target_correct | -0.5% | 10 | 12 | 0.8318 |
| final_answer_correct | +0.7% | 12 | 9 | 0.6636 |
| full_success | +0.5% | 10 | 8 | 0.8145 |

### feedback_grpo_vs_teacher_judge

| Metric | Difference | Improved | Regressed | McNemar p |
|---|---:|---:|---:|---:|
| tags_exact | +0.0% | 1 | 1 | 1.0000 |
| code_blocks_complete | +0.0% | 1 | 1 | 1.0000 |
| signature_exact | +0.0% | 1 | 1 | 1.0000 |
| python_executable | +0.3% | 5 | 4 | 1.0000 |
| all_hidden_tests_pass | -0.5% | 4 | 6 | 0.7539 |
| backward_target_correct | -0.5% | 7 | 9 | 0.8036 |
| final_answer_correct | -0.7% | 7 | 10 | 0.6291 |
| full_success | -0.8% | 5 | 8 | 0.5811 |

## Decision rule

Treat teacher guidance as confirmed only if the locked-set final-answer gain is at least 0.02, paired improvements exceed regressions, full_success does not decrease, Python executability decreases by at most 0.01, and the direction is not confined to one source or difficulty stratum. McNemar and Holm p-values are reported as uncertainty evidence, not as the sole criterion.

This is a custom executable FSG evaluation, not an official source-benchmark score.
