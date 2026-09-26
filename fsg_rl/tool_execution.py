"""Isolated execution of policy-generated Python and verifier tests."""

from __future__ import annotations

import ast
import functools
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .answer_equivalence import answers_equivalent
from .parsing import span_by_node
from .sandbox_safety import SandboxSafetyError, validate_python_code, validate_python_expression
from .schemas import ExecutionResult, FunctionGraph, NodeExecutionResult, ParsedFunctionSpan


class ToolExecutor:
    """Runs generated code with a configured isolation or local-limit backend."""

    def __init__(self, config: Dict[str, Any]):
        section = config.get("sandbox", {})
        self.backend = str(section.get("backend", "docker"))
        self.timeout_seconds = float(section.get("timeout_seconds", 5))
        self.docker_image = str(section.get("docker_image", "python:3.11-slim"))
        self.memory_limit = str(section.get("memory_limit", "512m"))
        self.memory_fraction = float(section.get("memory_fraction", 0.40))
        self.local_memory_max = str(section.get("local_memory_max", "8g"))
        self.cpu_limit = str(section.get("cpu_limit", "1.0"))
        self.pids_limit = int(section.get("pids_limit", 64))
        self.python_executable = str(section.get("python_executable", sys.executable))
        if self.backend == "docker" and shutil.which("docker") is None:
            raise RuntimeError("sandbox.backend='docker' requires the docker executable")
        if self.backend == "unshare" and shutil.which("unshare") is None:
            raise RuntimeError("sandbox.backend='unshare' requires the unshare executable")
        if not 0 < self.memory_fraction <= 1:
            raise ValueError("sandbox.memory_fraction must be in the interval (0, 1]")
        if self.backend in {"local_limited", "subprocess"} and not section.get(
            "allow_unsafe_subprocess", False
        ):
            raise ValueError(
                "Local host execution is unsafe; explicitly set "
                "sandbox.allow_unsafe_subprocess=true only for trusted tests"
            )

    def execute(self, parsed_spans: List[ParsedFunctionSpan], graph: FunctionGraph) -> ExecutionResult:
        node_results: Dict[str, NodeExecutionResult] = {}
        all_stdout: List[str] = []
        all_stderr: List[str] = []
        total_runtime = 0.0
        any_timeout = False
        executable_results: List[bool] = []

        for node in graph.nodes:
            expects_execution = bool(node.verification_spec.get("tests")) or (
                node.expected_output_type == "python_function"
            )
            if not expects_execution:
                continue
            span = span_by_node(parsed_spans, node.id)
            if not span or not span.code_blocks:
                result = NodeExecutionResult(
                    node_id=node.id,
                    executable=False,
                    stderr="missing Python code block",
                )
            else:
                result = self._execute_node(
                    node.id,
                    span.code_blocks[-1],
                    node.verification_spec,
                )
            node_results[node.id] = result
            all_stdout.append(result.stdout)
            all_stderr.append(result.stderr)
            total_runtime += result.runtime_seconds
            any_timeout = any_timeout or result.timeout
            executable_results.append(result.executable)

        return ExecutionResult(
            node_results=node_results,
            stdout="\n".join(part for part in all_stdout if part),
            stderr="\n".join(part for part in all_stderr if part),
            runtime_seconds=total_runtime,
            timeout=any_timeout,
            executable=all(executable_results) if executable_results else True,
        )

    def _execute_node(
        self,
        node_id: str,
        code: str,
        verification_spec: Dict[str, Any],
    ) -> NodeExecutionResult:
        tests = list(verification_spec.get("tests", []))
        if self.backend in {"unshare", "local_limited", "subprocess"}:
            try:
                code = validate_python_code(code, label=f"node {node_id!r} code")
                for index, test in enumerate(tests):
                    expression = str(test.get("call", test.get("expression", "")))
                    validate_python_expression(
                        expression,
                        label=f"node {node_id!r} test {index}",
                    )
            except SandboxSafetyError as exc:
                return NodeExecutionResult(
                    node_id=node_id,
                    executable=False,
                    stderr=f"static sandbox rejection: {exc}",
                )
        harness, marker = self._build_harness(code, tests)
        started = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="fsg-sandbox-") as temporary_dir:
            script_path = Path(temporary_dir) / "candidate.py"
            script_path.write_text(harness, encoding="utf-8")
            container_name = f"fsg-{uuid.uuid4().hex}" if self.backend == "docker" else None
            command, kwargs = self._command(
                script_path,
                Path(temporary_dir),
                container_name,
            )
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                    **kwargs,
                )
                runtime = time.monotonic() - started
            except subprocess.TimeoutExpired as exc:
                if container_name:
                    self._remove_container(container_name)
                return NodeExecutionResult(
                    node_id=node_id,
                    executable=False,
                    stdout=_decode_timeout_output(exc.stdout),
                    stderr=_decode_timeout_output(exc.stderr) or "timeout",
                    runtime_seconds=time.monotonic() - started,
                    timeout=True,
                )

        outputs_by_index = self._parse_outputs(completed.stdout, marker)
        outputs: Dict[str, Any] = {}
        test_results = []
        for index, test in enumerate(tests):
            expression = str(test.get("call", test.get("expression", "")))
            expected = test.get("expected", True if test.get("kind") == "property" else "")
            actual = outputs_by_index.get(index)
            passed = _values_equal(actual, expected)
            outputs[expression] = "" if actual is None else str(actual)
            test_results.append(
                {
                    "kind": str(test.get("kind", "unit")),
                    "expression": expression,
                    "expected": str(expected),
                    "actual": "" if actual is None else str(actual),
                    "passed": passed,
                }
            )

        return NodeExecutionResult(
            node_id=node_id,
            executable=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            runtime_seconds=runtime,
            timeout=False,
            outputs=outputs,
            test_results=test_results,
        )

    def _command(
        self,
        script_path: Path,
        directory: Path,
        container_name: str | None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        if self.backend == "docker":
            if not container_name:
                raise ValueError("Docker execution requires a unique container name")
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                str(self.pids_limit),
                "--memory",
                self.memory_limit,
                "--cpus",
                self.cpu_limit,
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "-v",
                f"{directory}:/workspace:ro",
                self.docker_image,
                "python3",
                "/workspace/candidate.py",
            ]
            return command, {}

        kwargs: Dict[str, Any] = {
            "cwd": str(directory),
            "env": {"PATH": os.environ.get("PATH", "")},
        }
        if sys.platform.startswith("linux"):
            memory_bytes = self._local_memory_limit_bytes()
            kwargs["preexec_fn"] = functools.partial(
                self._limit_subprocess,
                memory_bytes=memory_bytes,
            )
        if self.backend == "unshare":
            return [
                "unshare",
                "--user",
                "--map-root-user",
                "--net",
                "--pid",
                "--fork",
                "--mount-proc",
                self.python_executable,
                "-I",
                str(script_path),
            ], kwargs
        return [self.python_executable, "-I", str(script_path)], kwargs

    def _limit_subprocess(self, *, memory_bytes: int) -> None:
        cpu_seconds = max(1, int(self.timeout_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (self.pids_limit, self.pids_limit))

    def _local_memory_limit_bytes(self) -> int:
        if self.backend == "local_limited":
            proportional_limit = int(_available_memory_bytes() * self.memory_fraction)
            return max(64 * 1024 * 1024, min(
                proportional_limit,
                _parse_memory_bytes(self.local_memory_max),
            ))
        return _parse_memory_bytes(self.memory_limit)

    def local_resource_summary(self) -> Dict[str, Any]:
        """Return the limits that would be applied to a local child process."""
        available = _available_memory_bytes()
        return {
            "backend": self.backend,
            "available_memory_bytes": available,
            "memory_fraction": self.memory_fraction,
            "memory_limit_bytes": self._local_memory_limit_bytes(),
            "timeout_seconds": self.timeout_seconds,
            "pids_limit": self.pids_limit,
            "environment_keys": ["PATH"],
        }

    @staticmethod
    def _remove_container(container_name: str) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    @staticmethod
    def _build_harness(code: str, tests: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
        marker = f"FSG_RESULT_{uuid.uuid4().hex}"
        lines = [
            "import builtins as _fsg_builtins",
            "import collections as _fsg_collections",
            "import fractions as _fsg_fractions",
            "import functools as _fsg_functools",
            "import heapq as _fsg_heapq",
            "import itertools as _fsg_itertools",
            "import math as _fsg_math",
            "import operator as _fsg_operator",
            "import statistics as _fsg_statistics",
            "",
            f"_fsg_candidate_source = {code!r}",
            "_fsg_candidate_globals = {",
            "    '__builtins__': __builtins__,",
            "    '__name__': '__fsg_candidate__',",
            "}",
            "exec(compile(_fsg_candidate_source, '<candidate>', 'exec'), _fsg_candidate_globals)",
            "_fsg_test_globals = {",
            "    '__builtins__': __builtins__,",
            "    **{",
            "        name: value",
            "        for name, value in _fsg_candidate_globals.items()",
            "        if not name.startswith('__')",
            "    },",
            "}",
            "_fsg_allowed_modules = {",
            "    'collections': _fsg_collections,",
            "    'fractions': _fsg_fractions,",
            "    'functools': _fsg_functools,",
            "    'heapq': _fsg_heapq,",
            "    'itertools': _fsg_itertools,",
            "    'math': _fsg_math,",
            "    'operator': _fsg_operator,",
            "    'statistics': _fsg_statistics,",
            "}",
            "for _fsg_module_name, _fsg_module in _fsg_allowed_modules.items():",
            "    _fsg_test_globals.setdefault(_fsg_module_name, _fsg_module)",
            "    for _fsg_name in dir(_fsg_module):",
            "        if (",
            "            not _fsg_name.startswith('_')",
            "            and not hasattr(_fsg_builtins, _fsg_name)",
            "        ):",
            "            _fsg_test_globals.setdefault(_fsg_name, getattr(_fsg_module, _fsg_name))",
            "",
        ]
        for index, test in enumerate(tests):
            expression = str(test.get("call", test.get("expression", ""))).strip()
            if not expression:
                continue
            lines.extend(
                [
                    (
                        f"_fsg_value_{index} = eval(compile({expression!r}, "
                        f"'<hidden-test-{index}>', 'eval'), _fsg_test_globals)"
                    ),
                    f"print({marker!r} + '::{index}::' + repr(_fsg_value_{index}))",
                ]
            )
        return "\n".join(lines) + "\n", marker

    @staticmethod
    def _parse_outputs(stdout: str, marker: str) -> Dict[int, Any]:
        outputs: Dict[int, Any] = {}
        prefix = marker + "::"
        for line in stdout.splitlines():
            if not line.startswith(prefix):
                continue
            _, index, raw_value = line.split("::", 2)
            try:
                value = ast.literal_eval(raw_value)
            except Exception:
                value = raw_value
            outputs[int(index)] = value
        return outputs


def _values_equal(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _values_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    scalar_types = (str, int, float)
    if isinstance(actual, scalar_types) and isinstance(expected, scalar_types):
        return answers_equivalent(actual, expected)
    return False


def _decode_timeout_output(value: Any) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _available_memory_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except (OSError, ValueError):
        completed = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip().isdigit():
            return int(completed.stdout.strip())
        vm_stat = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if vm_stat.returncode == 0:
            lines = vm_stat.stdout.splitlines()
            page_size = 4096
            if lines and "page size of" in lines[0]:
                page_size = int(lines[0].split("page size of", 1)[1].split()[0])
            available_labels = {
                "Pages free",
                "Pages inactive",
                "Pages speculative",
                "Pages purgeable",
            }
            available_pages = 0
            for line in lines[1:]:
                label, separator, raw_value = line.partition(":")
                if separator and label in available_labels:
                    available_pages += int(raw_value.strip().rstrip("."))
            if available_pages:
                return available_pages * page_size
        raise RuntimeError("Unable to determine available system memory")


def _parse_memory_bytes(value: str | int) -> int:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("memory limit must be positive")
        return value
    raw = str(value).strip().lower()
    suffixes = {
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
    }
    for suffix in sorted(suffixes, key=len, reverse=True):
        if raw.endswith(suffix):
            number = raw[: -len(suffix)].strip()
            result = int(float(number) * suffixes[suffix])
            if result <= 0:
                break
            return result
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    raise ValueError(f"Invalid memory limit: {value!r}")
