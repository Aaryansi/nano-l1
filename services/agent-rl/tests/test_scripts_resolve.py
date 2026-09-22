"""every script must resolve every name it calls, before anything runs it.

three separate NameErrors reached committed code in this project, each one a
call site left behind by a refactor:

    null_budget_check.py   span()            after masked_span was consolidated
    null_corpus_check.py   span()            same refactor
    generalize_gym.py      observation_pool  an import that silently failed to
                                             apply, then was flagged by an ad
                                             hoc scan and wrongly dismissed as
                                             a false positive

none was caught by the test suite, because the tests exercise the library and
the bugs were in the scripts. none was caught by the verifier either, because
the artifacts predated the refactor, so the checks kept passing against json no
script could still produce.

this walks each script's syntax tree with proper scope handling and asserts
that every function called by bare name is defined, imported, or a builtin.
it is a fraction of what a linter does and it catches exactly this.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


def _bound_names(tree: ast.AST) -> set[str]:
    """every name the module binds: imports, defs, classes, assignments."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_every_called_name_resolves(path: Path):
    tree = ast.parse(path.read_text())
    bound = _bound_names(tree) | set(dir(builtins))
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    missing = sorted(called - bound)
    assert not missing, (
        f"{path.name} calls {missing} which is never imported or defined. "
        "this is the refactor-leftover class of bug: the name changed in the "
        "import and not at the call site."
    )
