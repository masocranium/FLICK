"""The documentation's examples have to actually run, and its links resolve."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def _python_blocks():
    for doc in DOCS:
        blocks = re.findall(r"```python\n(.*?)```", doc.read_text(), flags=re.S)
        for i, code in enumerate(blocks):
            yield pytest.param(code, id=f"{doc.name}-{i}")


@pytest.mark.parametrize("code", list(_python_blocks()))
def test_python_examples_run(code):
    exec(compile(code, "<doc example>", "exec"), {})


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_relative_links_resolve(doc):
    for target in re.findall(r"\]\((?!https?://)([^)#\s]+)", doc.read_text()):
        assert (doc.parent / target).exists(), f"{doc.name} links to missing {target}"
