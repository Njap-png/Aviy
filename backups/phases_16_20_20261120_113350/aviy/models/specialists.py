from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class SpecialistResult:
    specialist: str
    answer: str
    confidence: float
    evidence_required: bool = False
    metadata: dict[str, object] | None = None


class Specialist(Protocol):
    name: str
    domain: str

    def analyze(self, prompt: str) -> SpecialistResult: ...


class MathematicsSpecialist:
    name = "mathematics"
    domain = "mathematics"

    _binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }

    def evaluate_expression(self, expression: str) -> float:
        if not expression or len(expression) > 256:
            raise ValueError("Expression is empty or too long.")
        tree = ast.parse(expression, mode="eval")

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
                if isinstance(node.op, ast.Pow) and abs(right) > 12:
                    raise ValueError("Exponent exceeds the safety limit.")
                result = float(operation(left, right))
                if not math.isfinite(result) or abs(result) > 1e100:
                    raise ValueError("Expression result exceeds numeric limits.")
                return result
            raise ValueError("Expression contains an unsupported construct.")

        return visit(tree)

    def analyze(self, prompt: str) -> SpecialistResult:
        value = self.evaluate_expression(prompt.strip())
        return SpecialistResult(self.name, str(value), 1.0)


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
                metadata={"valid_syntax": False},
            )

        functions = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        for node in ast.walk(tree))
        classes = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        imports = sum(isinstance(node, (ast.Import, ast.ImportFrom))
                      for node in ast.walk(tree))
        return SpecialistResult(
            self.name,
            "Python syntax is valid; source was inspected without execution.",
            1.0,
            metadata={
                "valid_syntax": True,
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "executed": False,
            },
        )


class ScienceSpecialist:
    name = "science"
    domain = "science"

    def analyze(self, prompt: str) -> SpecialistResult:
        if not prompt.strip():
            raise ValueError("A scientific question is required.")
        return SpecialistResult(
            self.name,
            "This question requires evidence and scientific validation; "
            "no scientific conclusion has been established.",
            0.0,
            evidence_required=True,
            metadata={"validated_claim": False},
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
            "mathematics": ("calculate", "equation", "math", "solve", "integral"),
            "programming": ("code", "python", "program", "debug", "function"),
            "science": ("science", "physics", "biology", "chemistry", "hypothesis"),
        }
        selected = [
            specialist for domain, specialist in self._specialists.items()
            if any(word in text for word in keywords.get(domain, ()))
        ]
        return tuple(selected or self._specialists.values())

    def collaborate(self, prompt: str) -> tuple[SpecialistResult, ...]:
        return tuple(item.analyze(prompt) for item in self.route(prompt))
