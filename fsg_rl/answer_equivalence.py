"""Fail-closed equivalence checks for mathematical final answers.

The checker intentionally supports a conservative subset of LaTeX commonly
found in benchmark answers. Unsupported prose or syntax is never treated as
equivalent merely because a numerical probe happens to agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import math
import re
import signal
import threading
from typing import Any, Optional


_SYMBOLIC_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class EquivalenceResult:
    equivalent: bool
    method: str
    normalized_left: str
    normalized_right: str
    error: Optional[str] = None


def check_answer_equivalence(left: Any, right: Any) -> EquivalenceResult:
    """Compare two benchmark answers and report how equivalence was decided."""

    if left is None or right is None:
        return EquivalenceResult(False, "missing", str(left), str(right))

    left_text = str(left)
    right_text = str(right)
    try:
        left_text = _canonical_latex(left_text)
        right_text = _canonical_latex(right_text)
        if not left_text or not right_text:
            return EquivalenceResult(False, "empty", left_text, right_text)
        if left_text == right_text:
            return EquivalenceResult(True, "canonical_text", left_text, right_text)

        numeric = _numeric_equivalence(left_text, right_text)
        if numeric is not None:
            return EquivalenceResult(numeric, "numeric", left_text, right_text)

        complex_numeric = _complex_numeric_equivalence(left_text, right_text)
        if complex_numeric is not None:
            return EquivalenceResult(
                complex_numeric,
                "complex_numeric",
                left_text,
                right_text,
            )

        relation = _relation_equivalence(left_text, right_text)
        if relation is not None:
            return EquivalenceResult(relation, "relation", left_text, right_text)

        finite_set = _finite_set_equivalence(left_text, right_text)
        if finite_set is not None:
            return EquivalenceResult(finite_set, "finite_set", left_text, right_text)

        symbolic = _symbolic_equivalence(left_text, right_text)
    except Exception as exc:
        return EquivalenceResult(
            False,
            "unsupported",
            left_text,
            right_text,
            error=f"{type(exc).__name__}: {exc}",
        )
    return EquivalenceResult(symbolic, "symbolic", left_text, right_text)


def answers_equivalent(left: Any, right: Any) -> bool:
    return check_answer_equivalence(left, right).equivalent


def _canonical_latex(value: str) -> str:
    text = value.strip()
    text = _extract_explicit_sentence_final(text)
    text = _unwrap_boxed(text)
    text = text.strip().strip("$").strip().rstrip(".")
    text = text.replace("−", "-").replace("×", "*").replace("·", "*")
    text = text.replace("\\%", "%").replace("°", "degrees")
    text = text.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\(?:,|!|;|quad|qquad)", "", text)
    text = re.sub(r"\\(?:cdot|times)", "*", text)
    text = re.sub(r"\\(?:leq|le)", "<=", text)
    text = re.sub(r"\\(?:geq|ge)", ">=", text)
    text = re.sub(r"\\neq", "!=", text)
    text = re.sub(r"\\infty", "oo", text)
    text = re.sub(r"\\pi\b", "pi", text)
    text = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "degrees", text)
    text = re.sub(
        r"\bFraction\s*\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)",
        r"((\1)/(\2))",
        text,
    )
    text = re.sub(
        r"\\(?:mathbb|mathrm|operatorname|text)\s*\{([^{}]*)\}",
        r"\1",
        text,
    )
    text = _replace_two_argument_command(text, "binom", "binomial")
    text = _replace_fractions(text)
    text = _replace_one_argument_command(text, "sqrt", "sqrt")
    text = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", text)
    text = text.replace("\\lceil", "ceiling(").replace("\\rceil", ")")
    text = text.replace("\\lfloor", "floor(").replace("\\rfloor", ")")
    text = text.replace("\\{", "{").replace("\\}", "}")
    text = text.replace("~", "")
    text = re.sub(r"\s+", "", text)
    if text.startswith("{") and text.endswith("}") and not _has_top_level_comma(text[1:-1]):
        text = text[1:-1]
    return text.lower()


def _extract_explicit_sentence_final(text: str) -> str:
    """Extract only an explicitly stated inline-math answer at sentence end."""
    match = re.search(
        r"(?:\bis\b|=)\s*\\\(\s*([^()]+?)\s*\\\)\s*\.?\s*$",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else text


def _unwrap_boxed(text: str) -> str:
    stripped = text.strip()
    for command in ("\\boxed", "\\fbox"):
        prefix = command + "{"
        if stripped.startswith(prefix):
            content, end = _balanced_group(stripped, len(command))
            if end == len(stripped):
                return content
    return stripped


def _replace_fractions(text: str) -> str:
    command_re = re.compile(r"\\(?:dfrac|tfrac|frac)")
    while True:
        match = command_re.search(text)
        if not match:
            return text
        numerator, first_end = _command_argument(text, match.end())
        denominator, second_end = _command_argument(text, first_end)
        replacement = f"(({numerator})/({denominator}))"
        text = text[: match.start()] + replacement + text[second_end:]


def _replace_one_argument_command(text: str, command: str, function: str) -> str:
    pattern = re.compile(rf"\\{re.escape(command)}")
    while True:
        match = pattern.search(text)
        if not match:
            return text
        argument, end = _command_argument(text, match.end())
        text = text[: match.start()] + f"{function}({argument})" + text[end:]


def _replace_two_argument_command(text: str, command: str, function: str) -> str:
    pattern = re.compile(rf"\\{re.escape(command)}")
    while True:
        match = pattern.search(text)
        if not match:
            return text
        first, first_end = _command_argument(text, match.end())
        second, second_end = _command_argument(text, first_end)
        text = (
            text[: match.start()]
            + f"{function}({first},{second})"
            + text[second_end:]
        )


def _command_argument(text: str, cursor: int) -> tuple[str, int]:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        raise ValueError("missing LaTeX command argument")
    if text[cursor] == "{":
        return _balanced_group(text, cursor)
    if text[cursor] == "\\":
        match = re.match(r"\\[A-Za-z]+", text[cursor:])
        if not match:
            raise ValueError("invalid LaTeX command argument")
        return match.group(0), cursor + len(match.group(0))
    return text[cursor], cursor + 1


def _balanced_group(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        raise ValueError("expected opening brace")
    depth = 1
    cursor = start + 1
    while cursor < len(text) and depth:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        raise ValueError("unbalanced LaTeX braces")
    return text[start + 1 : cursor - 1], cursor


def _numeric_equivalence(left: str, right: str) -> Optional[bool]:
    left = _strip_simple_measurement_unit(left)
    right = _strip_simple_measurement_unit(right)
    try:
        left_value = float(left)
        right_value = float(right)
    except ValueError:
        return None
    return math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-9)


def _strip_simple_measurement_unit(text: str) -> str:
    """Normalize a scalar's thousands separators and recognized unit suffix."""
    number = (
        r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
        r"(?:e[+-]?\d+)?"
    )
    unit = (
        r"(?:%|mm|cm|km|m|mg|kg|g|ms|sec|secs|s|min|mins|hr|hrs|h|ml|l|"
        r"millimeters?|centimeters?|kilometers?|meters?|milligrams?|"
        r"kilograms?|grams?|milliseconds?|seconds?|minutes?|hours?|days?|"
        r"weeks?|months?|years?|milliliters?|liters?|inches?|feet|foot|"
        r"yards?|miles?|dollars?|cents?|euros?|pounds?|degrees?)"
    )
    match = re.fullmatch(
        rf"({number})(?:{unit}(?:\^\(?[123]\)?)?)?",
        text,
    )
    return match.group(1).replace(",", "") if match else text


def _complex_numeric_equivalence(left: str, right: str) -> Optional[bool]:
    """Compare strict numeric complex literals using either ``i`` or ``j``."""
    left_value = _parse_complex_literal(left)
    right_value = _parse_complex_literal(right)
    if left_value is None and right_value is None:
        return None
    if left_value is None or right_value is None:
        return False
    return math.isclose(
        left_value.real,
        right_value.real,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ) and math.isclose(
        left_value.imag,
        right_value.imag,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _parse_complex_literal(text: str) -> Optional[complex]:
    value = text
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]

    unsigned = r"(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
    rectangular = re.fullmatch(
        rf"(?P<real>[+-]?{unsigned})(?P<sign>[+-])"
        rf"(?P<imag>{unsigned})?[ij]",
        value,
    )
    if rectangular:
        real = float(rectangular.group("real"))
        magnitude = float(rectangular.group("imag") or "1")
        imag = magnitude if rectangular.group("sign") == "+" else -magnitude
        return complex(real, imag)

    imaginary = re.fullmatch(
        rf"(?P<sign>[+-]?)(?P<imag>{unsigned})?[ij]",
        value,
    )
    if imaginary:
        magnitude = float(imaginary.group("imag") or "1")
        if imaginary.group("sign") == "-":
            magnitude = -magnitude
        return complex(0.0, magnitude)
    return None


def _relation_equivalence(left: str, right: str) -> Optional[bool]:
    left_parts = _split_relations(left)
    right_parts = _split_relations(right)
    if left_parts is None and right_parts is None:
        return None
    if left_parts is None or right_parts is None or len(left_parts) != len(right_parts):
        return False
    for (left_expr, left_op), (right_expr, right_op) in zip(left_parts, right_parts):
        if left_op != right_op or not _symbolic_equivalence(left_expr, right_expr):
            return False
    return True


def _split_relations(text: str) -> Optional[list[tuple[str, str]]]:
    operators = []
    expressions = []
    depth = 0
    start = 0
    cursor = 0
    while cursor < len(text):
        character = text[cursor]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if depth == 0:
            operator = next(
                (candidate for candidate in ("<=", ">=", "!=", "=") if text.startswith(candidate, cursor)),
                None,
            )
            if operator:
                expressions.append(text[start:cursor])
                operators.append(operator)
                cursor += len(operator)
                start = cursor
                continue
        cursor += 1
    if not operators:
        return None
    expressions.append(text[start:])
    if any(not expression for expression in expressions):
        return None
    return [
        (expression, operators[index] if index < len(operators) else "")
        for index, expression in enumerate(expressions)
    ]


def _finite_set_equivalence(left: str, right: str) -> Optional[bool]:
    if not (left.startswith("{") and left.endswith("}")) and not (
        right.startswith("{") and right.endswith("}")
    ):
        return None
    if not (left.startswith("{") and left.endswith("}") and right.startswith("{") and right.endswith("}")):
        return False
    left_items = _split_top_level(left[1:-1], ",")
    right_items = _split_top_level(right[1:-1], ",")
    if len(left_items) != len(right_items):
        return False
    unmatched = list(right_items)
    for left_item in left_items:
        for index, right_item in enumerate(unmatched):
            if _symbolic_equivalence(left_item, right_item):
                unmatched.pop(index)
                break
        else:
            return False
    return True


def _has_top_level_comma(text: str) -> bool:
    return len(_split_top_level(text, ",")) > 1


def _split_top_level(text: str, separator: str) -> list[str]:
    parts = []
    depth = 0
    start = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == separator and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part for part in parts if part]


def _symbolic_equivalence(left: str, right: str) -> bool:
    with _symbolic_time_limit(_SYMBOLIC_TIMEOUT_SECONDS):
        left_expr = _safe_sympy_parse(left)
        right_expr = _safe_sympy_parse(right)
        import sympy

        return bool(sympy.simplify(left_expr - right_expr) == 0)


@contextmanager
def _symbolic_time_limit(seconds: float):
    """Bound SymPy work on Unix main threads; fail closed elsewhere."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("symbolic equivalence requires the main thread")
    if not hasattr(signal, "setitimer"):
        yield
        return

    def handle_timeout(_signum, _frame):
        raise TimeoutError(f"symbolic equivalence exceeded {seconds:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _safe_sympy_parse(text: str) -> Any:
    if not text or len(text) > 2048:
        raise ValueError("symbolic answer is empty or too long")
    text_without_decimals = re.sub(r"\d+\.\d+", "0", text)
    if (
        "__" in text
        or "." in text_without_decimals
        or re.search(r"[^A-Za-z0-9+\-*/^().,]", text)
    ):
        raise ValueError("unsupported characters in symbolic answer")

    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    allowed_functions = {
        "abs": sympy.Abs,
        "binomial": sympy.binomial,
        "ceiling": sympy.ceiling,
        "floor": sympy.floor,
        "sqrt": sympy.sqrt,
    }
    identifiers = set(re.findall(r"[A-Za-z][A-Za-z0-9]*", text))
    unknown = identifiers - set(allowed_functions) - {"oo", "pi"}
    local_dict = dict(allowed_functions)
    local_dict["oo"] = sympy.oo
    local_dict["pi"] = sympy.pi
    local_dict.update({name: sympy.Symbol(name) for name in unknown})
    global_dict = {
        "__builtins__": {},
        "Integer": sympy.Integer,
        "Float": sympy.Float,
        "Rational": sympy.Rational,
        "Symbol": sympy.Symbol,
    }
    return parse_expr(
        text,
        local_dict=local_dict,
        global_dict=global_dict,
        transformations=standard_transformations
        + (implicit_multiplication_application, convert_xor),
        evaluate=True,
    )
