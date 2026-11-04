"""Static guards for the Colab notebook (execution is covered by the manual smoke run)."""
import ast
import re
from pathlib import Path

import nbformat
import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks" / "chatlens_colab.ipynb"


@pytest.fixture(scope="module")
def nb():
    if not NB.exists():
        pytest.skip("notebook not present")
    return nbformat.read(NB, 4)


def test_valid_notebook_with_parsable_code(nb):
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type == "code":
            ast.parse(c.source)


def test_no_editable_install(nb):
    """Regression: `pip install -e` is not importable from the already-running Colab kernel."""
    for c in nb.cells:
        if c.cell_type == "code":
            assert not re.search(r"pip install[^\n]*\s-e\s", c.source)


def test_nothing_numeric_is_imported_before_the_install_cell(nb):
    """Regression: installing packages after numpy/pandas were imported can break the running kernel."""
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    idx = next(i for i, s in enumerate(code) if "pip install" in s and "chatlens" in s)
    for s in code[:idx]:
        assert not re.search(r"^\s*(import|from)\s+(numpy|pandas|scipy|sklearn|chatlens|matplotlib)\b", s, re.M)


def test_outputs_are_cleared_in_the_shipped_notebook(nb):
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")


def test_kaggle_cli_version_supports_token_flow():
    text = (NB.parents[1] / "pyproject.toml").read_text()
    assert 'kaggle = ["kaggle>=2.2"]' in text
