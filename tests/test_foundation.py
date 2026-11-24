import importlib
import pathlib
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGES = [
    "aviy",
    "aviy.core",
    "aviy.models",
    "aviy.models.architectures",
    "aviy.tokenizer",
    "aviy.data",
    "aviy.learning",
    "aviy.training",
    "aviy.cognition",
    "aviy.memory",
    "aviy.knowledge",
    "aviy.agents",
    "aviy.tools",
    "aviy.multimodal",
    "aviy.autonomous",
    "aviy.evaluation",
    "aviy.infrastructure",
    "aviy.security",
    "aviy.api",
    "aviy.cli",
    "aviy.math",
]


class FoundationTests(unittest.TestCase):
    def test_all_package_imports(self):
        for package in PACKAGES:
            with self.subTest(package=package):
                importlib.import_module(package)

    def test_project_metadata_is_valid(self):
        with (ROOT / "pyproject.toml").open("rb") as stream:
            metadata = tomllib.load(stream)
        self.assertEqual(metadata["project"]["name"], "aviy")
        self.assertEqual(metadata["project"]["requires-python"], ">=3.12,<3.13")

    def test_checklist_has_30_phases(self):
        content = (ROOT / "PROJECT_CHECKLIST.md").read_text(encoding="utf-8")
        phases = [line for line in content.splitlines() if line.startswith("## Phase ")]
        self.assertEqual(len(phases), 30)

    def test_verified_phase_one_and_two_entries_remain_checked(self):
        content = (ROOT / "PROJECT_CHECKLIST.md").read_text(encoding="utf-8")
        expected = [
            "- [x] Define package and module boundaries",
            "- [x] Configure Python 3.12, packaging and quality tools",
            "- [x] Establish documentation, logging and error conventions",
            "- [x] Define stable component interfaces",
            "- [x] Implement validated registries and dependency injection",
            "- [x] Implement coordination and lifecycle contracts",
        ]
        for entry in expected:
            self.assertIn(entry, content)

    def test_reusable_bootstrap_exists(self):
        self.assertTrue((ROOT / "scripts/bootstrap_aviy.ps1").is_file())
