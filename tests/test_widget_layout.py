"""Self-check: saved workflows keep loading.

ComfyUI restores a node's widget values by POSITION from ``widgets_values``.
A new input inserted in front of an existing one shifts every saved workflow
by one -- in 0.6.1 that put 50 into Voice Continue's text_top_p (max 1.0) and
ComfyUI rejected the node. Two checks hold the line:

* every node's widget order released on main (tests/released_widget_layouts.json)
  must stay a prefix of today's -- new inputs are appended, never inserted;
* every shipped example workflow must map onto today's widgets with each value
  inside its range and every dropdown value still offered.

Plain asserts, no pytest (see test_model_paths.py for why). Run:

    python tests/test_widget_layout.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
CONTROL = object()  # the frontend's control_after_generate slot after a seed


def _import_nodes():
    sys.modules.pop("folder_paths", None)
    spec = importlib.util.spec_from_file_location("moss_nodes_layout", ROOT / "nodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _widgets(cls) -> list[tuple[str, object]]:
    """``[(name, spec entry | CONTROL)]`` in the order the frontend builds them."""
    out: list[tuple[str, object]] = []
    for section in ("required", "optional"):
        for name, entry in cls.INPUT_TYPES().get(section, {}).items():
            typ = entry[0]
            opts = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            if opts.get("forceInput") or not (isinstance(typ, list) or typ in WIDGET_TYPES):
                continue
            out.append((name, entry))
            if typ == "INT" and (opts.get("control_after_generate") or name in ("seed", "noise_seed")):
                out.append((f"{name}:control", CONTROL))
    return out


def check_released_layouts_are_a_prefix(n) -> None:
    released = json.loads((ROOT / "tests" / "released_widget_layouts.json")
                          .read_text(encoding="utf-8"))["nodes"]
    assert released, "no released layouts recorded -- this check would be vacuous"
    for node, old in released.items():
        assert node in n.NODE_CLASS_MAPPINGS, f"{node} was released and is gone"
        now = [name for name, entry in _widgets(n.NODE_CLASS_MAPPINGS[node])
               if entry is not CONTROL]
        assert now[:len(old)] == old, (
            f"{node}: released widgets {old} are no longer a prefix of {now} -- "
            f"a new input was inserted instead of appended")


def check_example_workflows_fit_the_widgets(n) -> None:
    workflows = sorted((ROOT / "example_workflows").glob("*.json"))
    assert workflows, "no example workflows found -- this check would be vacuous"
    checked = 0
    for wf_path in workflows:
        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        for node in wf.get("nodes", []):
            cls = n.NODE_CLASS_MAPPINGS.get(node.get("type"))
            if cls is None:
                continue  # a core node -- not ours to check
            slots = _widgets(cls)
            values = node.get("widgets_values") or []
            where = f"{wf_path.name} node {node.get('id')} ({node['type']})"
            assert len(values) <= len(slots), (
                f"{where}: {len(values)} saved values for {len(slots)} widgets")
            for (name, entry), value in zip(slots, values):
                if entry is CONTROL:
                    continue
                typ, opts = entry[0], (entry[1] if len(entry) > 1 else {})
                if isinstance(typ, list):
                    assert value in typ, f"{where}: {name}={value!r} is not offered"
                elif typ in ("INT", "FLOAT"):
                    assert isinstance(value, (int, float)), f"{where}: {name}={value!r}"
                    lo, hi = opts.get("min"), opts.get("max")
                    assert lo is None or value >= lo, f"{where}: {name}={value} < min {lo}"
                    assert hi is None or value <= hi, f"{where}: {name}={value} > max {hi}"
            checked += 1
    assert checked, "no pack node found in any example workflow -- vacuous"


def main() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("check_")]
    n = _import_nodes()
    failed = 0
    for check in checks:
        try:
            check(n)
            print(f"  ok   {check.__name__}")
        except Exception as e:  # a check that explodes is a failed check
            failed += 1
            print(f"  FAIL {check.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
