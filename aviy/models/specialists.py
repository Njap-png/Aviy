from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol


@dataclass(frozen=True)
class SpecialistResult:
    specialist: str
    answer: str
    confidence: float
    evidence_required: bool = False
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between zero and one.")


class Specialist(Protocol):
    name: str
    domain: str

    def analyze(self, prompt: str) -> SpecialistResult: ...


@dataclass(frozen=True)
class BenchmarkCase:
    prompt: str
    expected_contains: str | None = None
    expected_exact: str | None = None
    expected_numeric: float | None = None
    required_metadata: dict[str, object] | None = None
    expected_confidence: float | None = None
    expected_evidence_required: bool | None = None
    absolute_tolerance: float = 1e-9
    relative_tolerance: float = 1e-9


@dataclass(frozen=True)
class BenchmarkCaseResult:
    index: int
    passed: bool
    answer: str
    reason: str


@dataclass(frozen=True)
class BenchmarkReport:
    domain: str
    passed: int
    total: int
    accuracy: float
    threshold: float
    accepted: bool
    case_results: tuple[BenchmarkCaseResult, ...] = ()


class SpecialistBenchmark:
    @staticmethod
    def _numeric_answer(result: SpecialistResult) -> float | None:
        metadata = result.metadata or {}

        value = metadata.get("result")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            return number if math.isfinite(number) else None

        solution = metadata.get("solution")
        if isinstance(solution, dict):
            value = solution.get("x")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                number = float(value)
                return number if math.isfinite(number) else None

        answer = result.answer.strip()
        if answer.casefold().startswith("x"):
            if "=" not in answer:
                return None
            answer = answer.split("=", 1)[1].strip()

        try:
            number = float(answer)
        except (TypeError, ValueError, OverflowError):
            return None

        return number if math.isfinite(number) else None

    @staticmethod
    def _validate_case(case: BenchmarkCase) -> None:
        if not isinstance(case, BenchmarkCase):
            raise TypeError("Every benchmark item must be a BenchmarkCase.")
        if not isinstance(case.prompt, str) or not case.prompt.strip():
            raise ValueError("Benchmark prompts must be non-empty strings.")

        has_criterion = any((
            case.expected_contains is not None,
            case.expected_exact is not None,
            case.expected_numeric is not None,
            case.required_metadata is not None,
            case.expected_confidence is not None,
            case.expected_evidence_required is not None,
        ))
        if not has_criterion:
            raise ValueError("Each benchmark case requires an evaluation criterion.")

        for name, tolerance in (
            ("absolute_tolerance", case.absolute_tolerance),
            ("relative_tolerance", case.relative_tolerance),
        ):
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance)
                or tolerance < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number.")

        if case.expected_numeric is not None and (
            isinstance(case.expected_numeric, bool)
            or not isinstance(case.expected_numeric, (int, float))
            or not math.isfinite(case.expected_numeric)
        ):
                raise ValueError("Expected numeric answers must be finite numbers.")

        if case.required_metadata is not None and not isinstance(
            case.required_metadata, dict
        ):
            raise ValueError("Required metadata must be a dictionary.")

        if case.expected_confidence is not None and (
            isinstance(case.expected_confidence, bool)
            or not isinstance(case.expected_confidence, (int, float))
            or not math.isfinite(case.expected_confidence)
            or not 0.0 <= case.expected_confidence <= 1.0
        ):
                raise ValueError("Expected confidence must be between zero and one.")

        if case.expected_evidence_required is not None and not isinstance(
            case.expected_evidence_required, bool
        ):
            raise ValueError("Expected evidence requirement must be boolean.")

    def evaluate(
        self,
        specialist: Specialist,
        cases: Sequence[BenchmarkCase],
        *,
        minimum_accuracy: float,
    ) -> BenchmarkReport:
        if (
            not isinstance(minimum_accuracy, (int, float))
            or isinstance(minimum_accuracy, bool)
            or not math.isfinite(minimum_accuracy)
            or not 0.0 <= minimum_accuracy <= 1.0
        ):
            raise ValueError("A valid finite accuracy threshold is required.")
        if not cases:
            raise ValueError("At least one benchmark case is required.")

        for case in cases:
            self._validate_case(case)

        case_results = []

        for index, case in enumerate(cases):
            try:
                result = specialist.analyze(case.prompt)
                if not isinstance(result, SpecialistResult):
                    raise TypeError("Specialist returned an invalid result.")

                failures = []
                answer = result.answer
                metadata = result.metadata or {}

                if case.expected_exact is not None:
                    actual = " ".join(answer.casefold().split())
                    expected = " ".join(case.expected_exact.casefold().split())
                    if actual != expected:
                        failures.append("Exact answer mismatch.")

                if (
                    case.expected_contains is not None
                    and case.expected_contains.casefold() not in answer.casefold()
                ):
                    failures.append("Expected text was not found.")

                if case.expected_numeric is not None:
                    actual_number = self._numeric_answer(result)
                    if actual_number is None:
                        failures.append("No valid numeric answer was produced.")
                    elif not math.isclose(
                        actual_number,
                        float(case.expected_numeric),
                        rel_tol=case.relative_tolerance,
                        abs_tol=case.absolute_tolerance,
                    ):
                        failures.append("Numeric answer is outside tolerance.")

                if case.required_metadata is not None:
                    for key, expected_value in case.required_metadata.items():
                        if key not in metadata:
                            failures.append(f"Missing metadata: {key}.")
                        elif metadata[key] != expected_value:
                            failures.append(f"Metadata mismatch: {key}.")

                if case.expected_confidence is not None and not math.isclose(
                    result.confidence,
                    case.expected_confidence,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    failures.append("Confidence mismatch.")

                if (
                    case.expected_evidence_required is not None
                    and result.evidence_required is not case.expected_evidence_required
                ):
                    failures.append("Evidence requirement mismatch.")

                passed = not failures
                reason = "All criteria passed." if passed else " ".join(failures)

                case_results.append(BenchmarkCaseResult(
                    index, passed, answer, reason
                ))

            except Exception as exc:  # noqa: BLE001 - isolate benchmark case failures
                case_results.append(BenchmarkCaseResult(
                    index,
                    False,
                    "",
                    f"Evaluation error: {type(exc).__name__}: {exc}",
                ))

        passed_count = sum(item.passed for item in case_results)
        accuracy = passed_count / len(cases)

        return BenchmarkReport(
            specialist.domain,
            passed_count,
            len(cases),
            accuracy,
            float(minimum_accuracy),
            accuracy >= minimum_accuracy,
            tuple(case_results),
        )


class MathematicsSpecialist:
    name = "mathematics"
    domain = "mathematics"

    _binary = MappingProxyType({
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    })

    def evaluate_expression(self, expression: str) -> float:
        if not expression or len(expression) > 256:
            raise ValueError("Expression is empty or too long.")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError("Invalid mathematical expression.") from exc

        def visit(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = float(node.value)
                if not math.isfinite(value):
                    raise ValueError("Non-finite constants are not allowed.")
                return value
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                value = visit(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp):
                operation = self._binary.get(type(node.op))
                if operation is None:
                    raise ValueError("Unsupported mathematical operator.")
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Pow) and (
                    abs(right) > 12 or abs(left) > 1e10
                ):
                    raise ValueError("Exponentiation exceeds safety limits.")
                try:
                    result = float(operation(left, right))
                except (ArithmeticError, OverflowError) as exc:
                    raise ValueError("Mathematical operation is undefined.") from exc
                if not math.isfinite(result) or abs(result) > 1e100:
                    raise ValueError("Expression result exceeds numeric limits.")
                return result
            raise ValueError("Expression contains an unsupported construct.")

        return visit(tree)

    def solve_linear_equation(self, equation: str) -> dict[str, float]:
        if not isinstance(equation, str) or len(equation) > 256:
            raise ValueError("Equation is invalid or too long.")
        if equation.count("=") != 1:
            raise ValueError("Provide one equation in the form ax+b=c.")

        left_text, right_text = (
            part.strip() for part in equation.split("=", 1)
        )
        if not left_text or not right_text:
            raise ValueError("Both sides of the equation are required.")

        def parse_linear(expression: str) -> tuple[float, float]:
            try:
                tree = ast.parse(expression.strip(), mode="eval")
            except SyntaxError as exc:
                raise ValueError("Invalid equation syntax.") from exc

            if len(list(ast.walk(tree))) > 128:
                raise ValueError("Equation is too complex.")

            def visit(node: ast.AST) -> tuple[float, float]:
                if isinstance(node, ast.Expression):
                    return visit(node.body)

                if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                    value = float(node.value)
                    if not math.isfinite(value) or abs(value) > 1e100:
                        raise ValueError("Numeric value exceeds safety limits.")
                    return 0.0, value

                if isinstance(node, ast.Name) and node.id == "x":
                    return 1.0, 0.0

                if isinstance(node, ast.UnaryOp) and isinstance(
                    node.op, (ast.UAdd, ast.USub)
                ):
                    a, b = visit(node.operand)
                    if isinstance(node.op, ast.USub):
                        return -a, -b
                    return a, b

                if isinstance(node, ast.BinOp):
                    left_a, left_b = visit(node.left)
                    right_a, right_b = visit(node.right)

                    if isinstance(node.op, ast.Add):
                        result = left_a + right_a, left_b + right_b
                    elif isinstance(node.op, ast.Sub):
                        result = left_a - right_a, left_b - right_b
                    elif isinstance(node.op, ast.Mult):
                        if left_a == 0:
                            result = right_a * left_b, right_b * left_b
                        elif right_a == 0:
                            result = left_a * right_b, left_b * right_b
                        else:
                            raise ValueError("Nonlinear multiplication is unsupported.")
                    elif isinstance(node.op, ast.Div):
                        if right_a != 0 or right_b == 0:
                            raise ValueError(
                                "Division requires a nonzero constant denominator."
                            )
                        result = left_a / right_b, left_b / right_b
                    else:
                        raise ValueError("Unsupported equation operator.")  # noqa: TRY004 - unsupported expression value

                    if any(not math.isfinite(value) or abs(value) > 1e100
                           for value in result):
                        raise ValueError("Equation exceeds numeric safety limits.")
                    return result

                raise ValueError("Equation contains an unsupported construct.")

            return visit(tree)

        left_a, left_b = parse_linear(left_text)
        right_a, right_b = parse_linear(right_text)

        coefficient = left_a - right_a
        constant = right_b - left_b

        if coefficient == 0:
            raise ValueError("Equation does not have exactly one solution.")

        solution = constant / coefficient
        if not math.isfinite(solution) or abs(solution) > 1e100:
            raise ValueError("Solution exceeds numeric safety limits.")

        return {"x": solution}

    def analyze(self, prompt: str) -> SpecialistResult:
        text = prompt.strip()
        for prefix in ("calculate ", "evaluate ", "solve "):
            if text.casefold().startswith(prefix):
                text = text[len(prefix):].strip()
                break
        if not text:
            raise ValueError("A mathematical problem is required.")
        if "=" in text:
            solution = self.solve_linear_equation(text)
            return SpecialistResult(
                self.name, f"x = {solution['x']}", 0.95,
                metadata={"method": "symbolic_equation_solver",
                          "solution": solution},
            )
        value = self.evaluate_expression(text)
        return SpecialistResult(
            self.name, str(value), 1.0,
            metadata={"method": "safe_arithmetic_evaluator", "result": value},
        )


class ProgrammingSpecialist:
    name = "programming"
    domain = "programming"

    def analyze(self, prompt: str) -> SpecialistResult:
        if not prompt.strip():
            raise ValueError("Source code is required.")
        try:
            tree = ast.parse(prompt)
        except SyntaxError as exc:
            return SpecialistResult(
                self.name,
                f"Syntax error at line {exc.lineno}: {exc.msg}",
                1.0,
                metadata={"valid_syntax": False, "executed": False},
            )

        nodes = list(ast.walk(tree))
        functions = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        for n in nodes)
        classes = sum(isinstance(n, ast.ClassDef) for n in nodes)
        imports = sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes)
        calls = [
            n.func.id if isinstance(n.func, ast.Name)
            else n.func.attr if isinstance(n.func, ast.Attribute) else "<dynamic>"
            for n in nodes if isinstance(n, ast.Call)
        ]
        dangerous = sorted(set(calls) & {
            "eval", "exec", "compile", "__import__", "input"
        })
        return SpecialistResult(
            self.name,
            "Python syntax is valid; static inspection completed without execution.",
            1.0,
            metadata={
                "valid_syntax": True,
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "calls": len(calls),
                "review_required_calls": dangerous,
                "executed": False,
                "analysis_scope": "static_python_ast",
            },
        )


class ScienceSpecialist:
    name = "science"
    domain = "science"

    def analyze(self, prompt: str) -> SpecialistResult:
        text = prompt.strip()
        if not text:
            raise ValueError("A scientific question is required.")
        lower = text.casefold()
        question_type = (
            "causal" if any(word in lower for word in ("cause", "causes", "effect"))
            else "hypothesis" if any(word in lower for word in ("hypothesis", "theory"))
            else "explanatory"
        )
        requirements = [
            "Identify the claim and define measurable terms.",
            "Find relevant primary research and independent corroboration.",
            "Check methodology, sample size, controls and limitations.",
            "Consider alternative explanations and possible falsification.",
        ]
        return SpecialistResult(
            self.name,
            "No scientific conclusion is established from the question alone. "
            "Evidence and independent validation are required.",
            0.0,
            evidence_required=True,
            metadata={
                "validated_claim": False,
                "question_type": question_type,
                "research_requirements": requirements,
                "evidence_status": "not_supplied",
            },
        )


class SpecialistRouter:
    def __init__(self, specialists: Sequence[Specialist]) -> None:
        if not specialists:
            raise ValueError("At least one specialist is required.")
        self._specialists = {item.domain: item for item in specialists}
        if len(self._specialists) != len(specialists):
            raise ValueError("Specialist domains must be unique.")

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(self._specialists)

    def route(self, prompt: str) -> tuple[Specialist, ...]:
        text = prompt.casefold()
        keywords = {
            "mathematics": ("calculate", "equation", "math", "solve", "integral",
                            "differentiate", "evaluate"),
            "programming": ("code", "python", "program", "debug", "function",
                            "syntax", "algorithm"),
            "science": ("science", "physics", "biology", "chemistry",
                        "hypothesis", "experiment", "scientific"),
        }
        selected = [
            specialist for domain, specialist in self._specialists.items()
            if any(word in text for word in keywords.get(domain, ()))
        ]
        return tuple(selected or self._specialists.values())

    def collaborate(self, prompt: str) -> tuple[SpecialistResult, ...]:
        results = []
        for specialist in self.route(prompt):
            try:
                results.append(specialist.analyze(prompt))
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                results.append(SpecialistResult(
                    specialist.name,
                    f"Unable to complete this analysis: {exc}",
                    0.0,
                    evidence_required=True,
                    metadata={"error": type(exc).__name__},
                ))
        return tuple(results)
