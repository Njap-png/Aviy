from pathlib import Path

import pytest

from aviy.core.errors import (
    AviyError,
    ConfigurationError,
    RecoveryError,
    ResourceLimitError,
    SecurityError,
    ValidationError,
)
from aviy.core.logging import get_logger

ROOT = Path(__file__).resolve().parents[1]


def test_engineering_standards_document_exists():
    document = ROOT / "docs" / "ENGINEERING_STANDARDS.md"
    assert document.is_file()
    content = document.read_text(encoding="utf-8")
    assert "Package boundaries" in content
    assert "Logging conventions" in content
    assert "Error conventions" in content


def test_all_application_packages_are_documented():
    document = (ROOT / "docs" / "ENGINEERING_STANDARDS.md").read_text(encoding="utf-8")
    packages = [
        "core",
        "models",
        "tokenizer",
        "data",
        "learning",
        "training",
        "cognition",
        "memory",
        "knowledge",
        "agents",
        "tools",
        "multimodal",
        "autonomous",
        "evaluation",
        "infrastructure",
        "security",
        "api",
        "cli",
    ]
    for package in packages:
        assert f"`{package}`" in document


def test_aviy_exception_hierarchy():
    exception_types = [
        ConfigurationError,
        ValidationError,
        ResourceLimitError,
        SecurityError,
        RecoveryError,
    ]
    assert all(issubclass(item, AviyError) for item in exception_types)


def test_logger_uses_aviy_namespace():
    logger = get_logger("tests.engineering")
    assert logger.name == "aviy.tests.engineering"


def test_logger_rejects_empty_name():
    with pytest.raises(ValueError):
        get_logger("")


def test_logger_rejects_whitespace_name():
    with pytest.raises(ValueError):
        get_logger("   ")
