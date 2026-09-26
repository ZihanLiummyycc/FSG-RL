from __future__ import annotations

import unittest
from unittest.mock import patch

from fsg_rl.sandbox_safety import SandboxSafetyError, validate_python_code
from fsg_rl.tool_execution import ToolExecutor, _parse_memory_bytes, _values_equal


class UnshareSandboxTests(unittest.TestCase):
    def test_rejects_file_access_and_dynamic_dunder(self):
        with self.assertRaises(SandboxSafetyError):
            validate_python_code("open('/etc/passwd').read()")
        with self.assertRaises(SandboxSafetyError):
            validate_python_code("getattr(1, '__class__')")

    @patch("fsg_rl.tool_execution.shutil.which", return_value="/usr/bin/unshare")
    def test_unshare_command_has_network_and_pid_namespaces(self, _which):
        executor = ToolExecutor(
            {
                "sandbox": {
                    "backend": "unshare",
                    "python_executable": "/usr/bin/python3",
                }
            }
        )
        command, _ = executor._command(
            script_path=__import__("pathlib").Path("/tmp/work/candidate.py"),
            directory=__import__("pathlib").Path("/tmp/work"),
            container_name=None,
        )
        self.assertIn("--net", command)
        self.assertIn("--pid", command)
        self.assertIn("--mount-proc", command)
        self.assertEqual(command[-2:], ["-I", "/tmp/work/candidate.py"])

    @patch("fsg_rl.tool_execution._available_memory_bytes", return_value=20 * 1024**3)
    def test_local_limited_uses_fraction_with_absolute_cap(self, _available):
        executor = ToolExecutor(
            {
                "sandbox": {
                    "backend": "local_limited",
                    "allow_unsafe_subprocess": True,
                    "memory_fraction": 0.40,
                    "local_memory_max": "6g",
                }
            }
        )
        self.assertEqual(executor._local_memory_limit_bytes(), 6 * 1024**3)
        self.assertEqual(executor.local_resource_summary()["environment_keys"], ["PATH"])

    @patch("fsg_rl.tool_execution._available_memory_bytes", return_value=10 * 1024**3)
    def test_local_limited_uses_available_memory_fraction(self, _available):
        executor = ToolExecutor(
            {
                "sandbox": {
                    "backend": "local_limited",
                    "allow_unsafe_subprocess": True,
                    "memory_fraction": 0.40,
                    "local_memory_max": "8g",
                }
            }
        )
        self.assertEqual(executor._local_memory_limit_bytes(), 4 * 1024**3)

    def test_memory_parser(self):
        self.assertEqual(_parse_memory_bytes("512m"), 512 * 1024**2)
        self.assertEqual(_parse_memory_bytes("8 GiB"), 8 * 1024**3)

    def test_json_lists_and_python_tuples_are_structurally_equal(self):
        self.assertTrue(_values_equal(((0, 0), (1, 0)), [[0, 0], [1, 0]]))
        self.assertFalse(_values_equal(((1, 0), (0, 0)), [[0, 0], [1, 0]]))

    def test_hidden_tests_have_independent_allowed_stdlib_symbols(self):
        executor = ToolExecutor(
            {
                "sandbox": {
                    "backend": "subprocess",
                    "allow_unsafe_subprocess": True,
                    "timeout_seconds": 2,
                    "memory_limit": "512m",
                }
            }
        )
        result = executor._execute_node(
            "f1",
            "import fractions\n\ndef numerator(value):\n    return value.numerator",
            {
                "tests": [
                    {
                        "kind": "unit",
                        "call": "numerator(Fraction(3, 2))",
                        "expected": 3,
                    }
                ]
            },
        )
        self.assertTrue(result.executable, result.stderr)
        self.assertTrue(result.test_results[0]["passed"])

    def test_hidden_test_prelude_does_not_mask_candidate_missing_import(self):
        executor = ToolExecutor(
            {
                "sandbox": {
                    "backend": "subprocess",
                    "allow_unsafe_subprocess": True,
                    "timeout_seconds": 2,
                    "memory_limit": "512m",
                }
            }
        )
        result = executor._execute_node(
            "f1",
            "def numerator(value):\n    return Fraction(value, 2).numerator",
            {
                "tests": [
                    {
                        "kind": "unit",
                        "call": "numerator(3)",
                        "expected": 3,
                    }
                ]
            },
        )
        self.assertFalse(result.executable)
        self.assertIn("Fraction", result.stderr)

    def test_hidden_test_stdlib_aliases_do_not_override_builtin_pow(self):
        executor = ToolExecutor(
            {
                "sandbox": {
                    "backend": "subprocess",
                    "allow_unsafe_subprocess": True,
                    "timeout_seconds": 2,
                    "memory_limit": "512m",
                }
            }
        )
        result = executor._execute_node(
            "f1",
            "def modular_power():\n    return pow(2, 3, 5)",
            {
                "tests": [
                    {
                        "kind": "property",
                        "expression": "pow(2, 3, 5) == 3 and modular_power() == 3",
                        "expected": True,
                    }
                ]
            },
        )
        self.assertTrue(result.executable, result.stderr)
        self.assertTrue(result.test_results[0]["passed"])


if __name__ == "__main__":
    unittest.main()
