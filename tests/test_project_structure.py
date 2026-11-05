"""
Teste për strukturën e projektit: verifikon që të gjitha paketat
dhe modulet jo-UI importohen pa gabime nga rrënja e projektit.
"""

import importlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NON_UI_MODULES = [
    "tasks",
    "tasks.social_golfers",
    "tasks.sudoku",
    "tasks.latin_square",
    "tasks.search_trees",
    "tasks.search_trees.eight_puzzle",
    "tasks.search_trees.missionaries_cannibals",
    "tasks.search_trees.workforce_scheduling",
    "utils",
    "utils.visualization",
    "utils.statistics",
    "utils.validation",
    "data",
    "data.sudoku_puzzles",
]

REQUIRED_FILES = [
    "app.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    "ui/dashboard.py",
    "ui/social_golfers_ui.py",
    "ui/sudoku_ui.py",
    "ui/latin_square_ui.py",
    "ui/search_trees_ui.py",
]


@pytest.mark.parametrize("module_name", NON_UI_MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("relative_path", REQUIRED_FILES)
def test_required_file_exists(relative_path):
    assert (PROJECT_ROOT / relative_path).is_file()


def test_requirements_contain_core_dependencies():
    content = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("streamlit", "matplotlib", "networkx", "pytest"):
        assert package in content
