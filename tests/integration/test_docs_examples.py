"""Guard the docs against rot: every ```python block must parse.

This is a syntax-level check — it compiles each fenced Python block in the
README and the ``docs/`` guides so a rename or a typo in an example fails
CI instead of shipping a broken snippet. (Semantic correctness of the
public API the examples call is covered by the unit suite + mypy.)
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = [_ROOT / "README.md", _ROOT / "docs" / "cookbook.md", _ROOT / "docs" / "custom_backend.md"]
_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for doc in _DOCS:
        text = doc.read_text(encoding="utf-8")
        for i, m in enumerate(_BLOCK.finditer(text)):
            found.append((doc.name, i, m.group(1)))
    return found


def test_docs_have_examples() -> None:
    # Sanity: if the regex ever stops matching, we'd silently test nothing.
    assert len(_blocks()) >= 10


@pytest.mark.parametrize("doc,idx,code", _blocks(), ids=lambda v: v if isinstance(v, str) else None)
def test_python_block_parses(doc: str, idx: int, code: str) -> None:
    try:
        ast.parse(code)
    except SyntaxError as err:  # pragma: no cover - failure path
        pytest.fail(f"{doc} python block #{idx} has a syntax error: {err}")
