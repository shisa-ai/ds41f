"""Every path that builds a model for inference must apply the expert placement.

The defect this guards: expert_placement.maybe_apply was called by the benchmark and
the profilers but not by serve/server.py or generate.py, so normal serving silently
kept contiguous expert ownership and lost the balanced placement's ~15% on 8K
prefill. A unit test cannot load the real TP4 model, so this parses the loaders and
asserts the call is present.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODEL_REPO = REPO.parent / "glm-testing" / "ds41f"

# loader file -> function that builds and loads the model
LOADERS = {
    MODEL_REPO / "serve" / "server.py": "load_model_rank",
    MODEL_REPO / "inference" / "generate.py": "main",
}


def _calls_maybe_apply(fn: ast.FunctionDef) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "maybe_apply":
                return True
    return False


@pytest.mark.parametrize("path,func", sorted(LOADERS.items(), key=lambda kv: str(kv[0])))
def test_model_loader_applies_expert_placement(path: Path, func: str):
    if not path.exists():
        pytest.skip(f"{path} not present in this checkout")
    tree = ast.parse(path.read_text())
    targets = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func]
    assert targets, f"{path}: no function named {func}"
    assert any(_calls_maybe_apply(f) for f in targets), (
        f"{path}:{func} builds a model but never calls expert_placement.maybe_apply; "
        "serving would keep contiguous expert ownership"
    )
