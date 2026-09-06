"""Self-check for the local model folders (issue #2).

The pack registers a ``moss_tts`` model folder with ComfyUI so a model that is
already on disk can be used without the Hugging Face cache -- either from
``ComfyUI/models/moss_tts/`` (or wherever ``extra_model_paths.yaml`` points
that name) or from an explicit ``model_path`` on the loader node.

Plain asserts, no pytest: the pack root is a Python package whose
``__init__.py`` only imports inside ComfyUI, which makes pytest's package
collection choke on it. Run it directly, from anywhere:

    python tests/test_model_paths.py

Needs only what ComfyUI already ships (torch/torchaudio, imported by nodes.py).
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import tempfile
import types
from pathlib import Path

NODES_PY = Path(__file__).resolve().parents[1] / "nodes.py"


def _import_nodes(*, keep_folder_paths: bool = False):
    """Load nodes.py by PATH, not by name.

    Two reasons: the pack root has an ``__init__.py`` (a plain import would
    make it a package import), and inside ComfyUI the name ``nodes`` is
    ComfyUI's OWN top-level module.
    """
    if not keep_folder_paths:
        sys.modules.pop("folder_paths", None)
    spec = importlib.util.spec_from_file_location("moss_nodes_selfcheck", NODES_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub_folder_paths(roots: list[str], models_dir: str = "/comfy/models"):
    """A minimal stand-in for ComfyUI's folder_paths module."""
    mod = types.ModuleType("folder_paths")
    mod.models_dir = models_dir
    mod.folder_names_and_paths = {}
    mod.get_folder_paths = lambda name: list(roots)
    mod.add_model_folder_path = lambda name, path, *a, **kw: roots.append(path)
    sys.modules["folder_paths"] = mod
    return mod


def _model_dir(root: Path, name: str, *, with_config: bool = True,
               model_type: str = "moss_tts_local",
               sampling_rate: int | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if with_config:
        fields = []
        if model_type is not None:
            fields.append('"model_type": "%s"' % model_type)
        if sampling_rate is not None:
            fields.append('"sampling_rate": %d' % sampling_rate)
        (d / "config.json").write_text("{%s}" % ", ".join(fields), encoding="utf-8")
    return d


def _capture_log() -> list[str]:
    """Collect what nodes.py logs from here on."""
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    log = logging.getLogger("MOSS-TTS-ComfyUI")
    # Without this the logger inherits root's WARNING and an assertion about
    # an info() message would pass without ever seeing one.
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    return records


def _fake_transformers():
    """A transformers stand-in whose from_pretrained stops the load at once."""
    class _Stop(Exception):
        pass

    class _Processor:
        @staticmethod
        def from_pretrained(load_id, **kw):
            raise _Stop()

    mod = types.ModuleType("transformers")
    mod.AutoProcessor = _Processor
    mod.AutoModel = _Processor
    return mod, _Stop


def _restore(name: str, saved) -> None:
    """Put a stubbed module back the way it was -- popping would UNLOAD a real
    one that a later check needs."""
    if saved is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = saved


def _raises(fn, *, contains: str) -> str:
    """Run fn, require a RuntimeError whose message contains `contains`."""
    try:
        fn()
    except RuntimeError as e:
        assert contains in str(e), f"expected {contains!r} in: {e}"
        return str(e)
    raise AssertionError(f"expected a RuntimeError containing {contains!r}")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_discovery(n, tmp: Path) -> None:
    _model_dir(tmp, "MOSS-8B-local")
    _model_dir(tmp, "just-a-folder", with_config=False)
    # A config.json that says nothing is still a model -- tolerance is
    # deliberate, model_type is only read to recognise the tokenizer.
    _model_dir(tmp, "odd-but-loadable", model_type=None)
    # ... and so is one whose config.json cannot be read at all: an odd
    # config must never make a model vanish from the dropdown.
    broken = _model_dir(tmp, "broken-config")
    (broken / "config.json").write_text("{not json", encoding="utf-8")
    # ... and the audio tokenizer, which a full offline set has lying right
    # next to the model, must NOT be offered as a TTS model.
    _model_dir(tmp, "MOSS-Audio-Tokenizer-v2", model_type="moss-audio-tokenizer")
    _stub_folder_paths([str(tmp)])

    found = n._local_model_dirs()
    assert found == {
        "local: MOSS-8B-local": str(tmp / "MOSS-8B-local"),
        "local: odd-but-loadable": str(tmp / "odd-but-loadable"),
        "local: broken-config": str(tmp / "broken-config"),
    }, found
    choices = n._model_choices()
    assert choices[: len(n.AVAILABLE_MODELS)] == n.AVAILABLE_MODELS, choices
    assert "local: MOSS-8B-local" in choices, choices


def check_missing_root_is_not_an_error(n, tmp: Path) -> None:
    # models/moss_tts does not exist until someone creates it.
    _stub_folder_paths([str(tmp / "nope")])
    assert n._local_model_dirs() == {}


def check_without_comfyui(n, tmp: Path) -> None:
    # None, not pop: run from a ComfyUI root the REAL folder_paths is
    # importable, and the assertions below would then see real models.
    sys.modules["folder_paths"] = None
    assert n._local_model_dirs() == {}
    assert n._model_choices() == n.AVAILABLE_MODELS

    # And quietly: nodes.py is imported outside ComfyUI too (the Comfy
    # registry's AST parser, this self-check). "could not register the model
    # folder" would be a warning about a situation that is perfectly normal.
    records = []

    class _Catch(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Catch()
    logger = logging.getLogger("MOSS-TTS-ComfyUI")
    logger.addHandler(handler)
    try:
        n._register_model_folder()
    finally:
        logger.removeHandler(handler)
    assert not records, [r.getMessage() for r in records]


def check_name_clash_first_root_wins(n, tmp: Path) -> None:
    first, second = tmp / "a", tmp / "b"
    _model_dir(first, "same")
    _model_dir(second, "same")
    _stub_folder_paths([str(first), str(second)])
    # Same precedence folder_paths itself uses.
    assert n._local_model_dirs()["local: same"] == str(first / "same")


def check_hub_label_keeps_its_repo_id(n, tmp: Path) -> None:
    assert n._repo_id_from_label(
        "OpenMOSS-Team/MOSS-TTS-v1.5 (8B)") == "OpenMOSS-Team/MOSS-TTS-v1.5"


def check_local_label_resolves_to_its_directory(n, tmp: Path) -> None:
    d = _model_dir(tmp, "my model (final)")
    _stub_folder_paths([str(tmp)])
    # Looked up, never parsed out of the label: a real path may contain " (".
    assert n._repo_id_from_label("local: my model (final)") == str(d)


def check_vanished_local_model_says_so(n, tmp: Path) -> None:
    _stub_folder_paths([str(tmp)])
    _raises(lambda: n._repo_id_from_label("local: weg"),
            contains="no longer under any")


def check_model_path_beats_the_dropdown(n, tmp: Path) -> None:
    # Compared against the RESOLVED path throughout: %TEMP% may be an 8.3
    # short name, which production canonicalises away (see below).
    d = _model_dir(tmp, "elsewhere").resolve()
    assert n._resolve_model_ref(n.DEFAULT_MODEL_LABEL, str(d)) == str(d)
    # Empty / whitespace leaves the dropdown alone.
    assert n._resolve_model_ref(
        "OpenMOSS-Team/MOSS-TTS-v1.5 (8B)", "   ") == "OpenMOSS-Team/MOSS-TTS-v1.5"
    # Quotes survive a copy-paste from the file explorer.
    assert n._resolve_model_ref("x", f'"{d}"') == str(d)
    # Canonical: _load_bundle caches by this string, so the same folder
    # written two ways must not load the model twice into VRAM.
    detour = d.parent / "elsewhere" / ".." / "elsewhere"
    assert n._resolve_model_ref("x", str(detour)) == str(d)


def check_model_path_rejects_a_file(n, tmp: Path) -> None:
    f = tmp / "model.safetensors"
    f.write_bytes(b"")
    _raises(lambda: n._resolve_model_ref("x", str(f)), contains="not a directory")


def check_model_path_without_config_names_what_it_found(n, tmp: Path) -> None:
    # One level too high is the usual mistake -- the error has to help.
    parent = tmp / "snapshots"
    _model_dir(parent, "abc123")
    msg = _raises(lambda: n._resolve_model_ref("x", str(parent)),
                  contains="no config.json")
    assert "abc123" in msg, msg


def check_registration(n, tmp: Path) -> None:
    roots: list[str] = []
    _stub_folder_paths(roots, models_dir=str(tmp / "models"))
    n._register_model_folder()
    assert roots == [str(tmp / "models" / "moss_tts")], roots


def check_a_broken_host_costs_the_dropdown_not_the_pack(n, tmp: Path) -> None:
    mod = _stub_folder_paths([])
    del mod.models_dir  # a host we do not understand
    n._register_model_folder()  # must not raise


def check_the_loader_node_offers_both(n, tmp: Path) -> None:
    """What ComfyUI actually calls. The dropdown is built per request, so a
    model copied in while ComfyUI runs shows up on a UI refresh."""
    _model_dir(tmp, "fresh-copy")
    _stub_folder_paths([str(tmp)])
    spec = n.MOSSLoadModel.INPUT_TYPES()
    assert "local: fresh-copy" in spec["required"]["model_id"][0]
    assert n.DEFAULT_MODEL_LABEL in spec["required"]["model_id"][0]
    assert spec["optional"]["model_path"][0] == "STRING"
    assert spec["optional"]["model_path"][1]["default"] == ""




def main() -> int:
    # Collected here, not at module level: a check defined further down the
    # file would silently not run.
    checks = [v for k, v in sorted(globals().items()) if k.startswith("check_")]
    n = _import_nodes()
    failed = 0
    for check in checks:
        with tempfile.TemporaryDirectory() as td:
            try:
                check(n, Path(td))
                print(f"  ok   {check.__name__}")
            except Exception as e:
                # Not just AssertionError: a check that EXPLODES is a failed
                # check, not a crashed runner -- otherwise a broken guard
                # (an uncaught OSError, a KeyError) reads as "no output".
                failed += 1
                print(f"  FAIL {check.__name__}: {type(e).__name__}: {e}")
            finally:
                sys.modules.pop("folder_paths", None)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0




def check_the_tokenizer_is_found_for_an_offline_load(n, tmp: Path) -> None:
    """A local model alone still pulls its audio tokenizer (7-8.5 GB) off the
    Hub. One lying in the same model folder is used instead, unasked."""
    model = _model_dir(tmp, "MOSS-TTS-v1.5", model_type="moss_tts_delay",
                       sampling_rate=24000)
    tok = _model_dir(tmp, "MOSS-Audio-Tokenizer",
                     model_type="moss-audio-tokenizer", sampling_rate=24000)
    _stub_folder_paths([str(tmp)])
    assert n._resolve_tokenizer_ref(str(model), "") == str(tok)


def check_a_hub_model_keeps_its_own_tokenizer(n, tmp: Path) -> None:
    """Nothing is saved by redirecting it, and the Hub path stays untouched --
    quietly: a Hub model has no readable sample rate, so without the isdir
    guard every single load would warn about a mismatch that is not one."""
    _model_dir(tmp, "MOSS-Audio-Tokenizer", model_type="moss-audio-tokenizer",
               sampling_rate=24000)
    _stub_folder_paths([str(tmp)])
    records = _capture_log()
    assert n._resolve_tokenizer_ref("OpenMOSS-Team/MOSS-TTS-v1.5", "") is None
    assert not records, records


def check_an_explicit_tokenizer_path_wins_and_is_checked(n, tmp: Path) -> None:
    model = _model_dir(tmp, "m", model_type="moss_tts_local", sampling_rate=48000)
    _model_dir(tmp, "beside", model_type="moss-audio-tokenizer",
               sampling_rate=48000)
    chosen = _model_dir(tmp / "elsewhere", "tok",
                        model_type="moss-audio-tokenizer", sampling_rate=48000)
    _stub_folder_paths([str(tmp)])
    assert n._resolve_tokenizer_ref(str(model), str(chosen)) == str(chosen)
    _raises(lambda: n._resolve_tokenizer_ref(str(model), str(tmp / "nix")),
            contains="no config.json")


def check_the_codec_path_reaches_the_processor(n, tmp: Path) -> None:
    """It is the whole point: MOSS's processor prefers codec_path over the
    repo id in processor_config.json."""
    seen = {}

    class _FakeProcessor:
        @staticmethod
        def from_pretrained(load_id, **kw):
            seen.update(load_id=load_id, **kw)
            raise _Stop()

    class _Stop(Exception):
        pass

    fake = types.ModuleType("transformers")
    fake.AutoProcessor = _FakeProcessor
    fake.AutoModel = _FakeProcessor
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = fake
    try:
        try:
            n._load_bundle("some/model", "cpu", "sdpa", codec_path=str(tmp))
        except _Stop:
            pass
        assert seen.get("codec_path") == str(tmp), seen
        # ... and without one, the kwarg is not sent at all (the model decides).
        seen.clear()
        try:
            n._load_bundle("some/model2", "cpu", "sdpa")
        except _Stop:
            pass
    finally:
        _restore("transformers", saved)
    # seen must not be empty here, or the assertion below would hold vacuously.
    assert seen.get("load_id") == "some/model2", seen
    assert "codec_path" not in seen, seen


def check_the_pack_registers_itself_on_import(n, tmp: Path) -> None:
    """The module-level _register_model_folder() call is the entire hook:
    without it nothing registers 'moss_tts', the dropdown never shows a local
    model, and extra_model_paths.yaml has no name to bind to."""
    roots: list[str] = []
    _stub_folder_paths(roots, models_dir=str(tmp / "models"))
    _import_nodes(keep_folder_paths=True)
    assert roots == [str(tmp / "models" / "moss_tts")], roots


def check_the_loader_node_wires_both_paths(n, tmp: Path) -> None:
    """load() is the only place the two inputs meet -- swapping the two
    resolvers there would be invisible to every other check."""
    model = _model_dir(tmp, "m", model_type="moss_tts_local", sampling_rate=48000)
    tok = _model_dir(tmp, "tok", model_type="moss-audio-tokenizer",
                     sampling_rate=48000)
    _stub_folder_paths([])
    seen: dict = {}
    saved = n._load_bundle
    n._load_bundle = lambda mid, dev, **kw: seen.update(
        model=mid, device=dev, **kw) or {"loaded": True}
    try:
        (bundle,) = n.MOSSLoadModel().load(
            "OpenMOSS-Team/MOSS-TTS-v1.5 (8B)", "cpu", "auto",
            str(model), str(tok))
    finally:
        n._load_bundle = saved
    assert bundle == {"loaded": True}, bundle
    assert seen.get("model") == str(model.resolve()), seen
    assert seen.get("codec_path") == str(tok.resolve()), seen


def check_the_tokenizer_is_found_beside_the_model(n, tmp: Path) -> None:
    """The model_path route points at a model ComfyUI knows nothing about --
    its tokenizer is then a sibling, which no registered root ever sees."""
    outside = tmp / "AI" / "models"
    model = _model_dir(outside, "MOSS-TTS-v1.5", model_type="moss_tts_delay",
                       sampling_rate=24000)
    tok = _model_dir(outside, "MOSS-Audio-Tokenizer",
                     model_type="moss-audio-tokenizer", sampling_rate=24000)
    _stub_folder_paths([])  # nothing registered at all
    assert n._resolve_tokenizer_ref(str(model), "") == str(tok)


def check_a_tokenizer_beside_the_model_beats_one_in_a_root(n, tmp: Path) -> None:
    """model_path is the "I know where it is" input, so its neighbour is the
    more specific answer than a registered root that happens to hold another
    tokenizer of the same rate -- which may be a different one."""
    outside = tmp / "library"
    model = _model_dir(outside, "m", model_type="moss_tts_local", sampling_rate=48000)
    beside = _model_dir(outside, "tok", model_type="moss-audio-tokenizer",
                        sampling_rate=48000)
    root = tmp / "comfy"
    _model_dir(root, "MOSS-Audio-Tokenizer-v2",
               model_type="moss-audio-tokenizer", sampling_rate=48000)
    _stub_folder_paths([str(root)])
    assert n._resolve_tokenizer_ref(str(model), "") == str(beside)


def check_the_tokenizer_is_matched_by_sample_rate(n, tmp: Path) -> None:
    """The 1.7B runs at 48 kHz with Audio-Tokenizer-v2, the 8B at 24 kHz with
    Audio-Tokenizer. Same quantiser count and codebook size, so the wrong one
    decodes WITHOUT a shape error -- straight to noise. Whoever keeps both
    models offline has both tokenizers in that folder, and "MOSS-Audio-
    Tokenizer" sorts before "...-v2", so picking the first would hand the
    48 kHz model the 24 kHz codec."""
    small = _model_dir(tmp, "MOSS-TTS-Local-Transformer-v1.5",
                       model_type="moss_tts_local", sampling_rate=48000)
    big = _model_dir(tmp, "MOSS-TTS-v1.5", model_type="moss_tts_delay",
                     sampling_rate=24000)
    v1 = _model_dir(tmp, "MOSS-Audio-Tokenizer",
                    model_type="moss-audio-tokenizer", sampling_rate=24000)
    v2 = _model_dir(tmp, "MOSS-Audio-Tokenizer-v2",
                    model_type="moss-audio-tokenizer", sampling_rate=48000)
    _stub_folder_paths([str(tmp)])
    assert n._resolve_tokenizer_ref(str(small), "") == str(v2)
    assert n._resolve_tokenizer_ref(str(big), "") == str(v1)


def check_a_tokenizer_of_the_wrong_rate_is_not_used(n, tmp: Path) -> None:
    """Better the Hub download the user wanted to avoid than silent noise."""
    model = _model_dir(tmp, "m", model_type="moss_tts_local", sampling_rate=48000)
    _model_dir(tmp, "MOSS-Audio-Tokenizer", model_type="moss-audio-tokenizer",
               sampling_rate=24000)
    _stub_folder_paths([str(tmp)])
    records = _capture_log()
    assert n._resolve_tokenizer_ref(str(model), "") is None
    assert any("matches" in r for r in records), records


def check_an_explicit_tokenizer_of_the_wrong_rate_is_not_silent(n, tmp: Path) -> None:
    """An explicit path is an explicit decision, so it is honoured -- but this
    combination produces noise, and that must not pass without a word."""
    model = _model_dir(tmp, "m", model_type="moss_tts_local", sampling_rate=48000)
    tok = _model_dir(tmp, "tok", model_type="moss-audio-tokenizer",
                     sampling_rate=24000)
    _stub_folder_paths([str(tmp)])
    records = _capture_log()
    assert n._resolve_tokenizer_ref(str(model), str(tok)) == str(tok)
    assert any("24000" in r and "48000" in r for r in records), records


def check_the_codec_path_is_part_of_the_model_cache_key(n, tmp: Path) -> None:
    """Two tokenizers, one model: without codec_path in the key the second
    load returns the first bundle -- the wrong codec, straight from VRAM."""
    attn = n._resolve_attention("cpu", "sdpa")
    sentinel = {"marker": "cached"}
    n._MODEL_CACHE[("m", "cpu", attn)] = sentinel        # the pre-fix key shape
    n._MODEL_CACHE[("m", "cpu", attn, None)] = sentinel  # today's, no codec
    fake, stop = _fake_transformers()
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = fake
    try:
        assert n._load_bundle("m", "cpu", "sdpa") is sentinel  # cache works
        try:
            n._load_bundle("m", "cpu", "sdpa", codec_path=str(tmp))
        except stop:
            pass
        else:
            raise AssertionError("a different codec_path reused the cached bundle")
    finally:
        n._MODEL_CACHE.clear()
        _restore("transformers", saved)


def check_a_local_folder_is_not_sent_to_snapshot_download(n, tmp: Path) -> None:
    """The Windows workaround resolves a REPO ID to a local snapshot. A local
    folder that fails with the same message is a real error -- handing it to
    snapshot_download would hide it behind a network call."""
    called: list[str] = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda mid, *a, **kw: called.append(mid) or str(tmp)

    class _Refuses:
        @staticmethod
        def from_pretrained(load_id, **kw):
            raise OSError("Repo id must use alphanumeric chars, got " + load_id)

    fake = types.ModuleType("transformers")
    fake.AutoProcessor = _Refuses
    fake.AutoModel = _Refuses
    saved = {k: sys.modules.get(k) for k in ("transformers", "huggingface_hub")}
    sys.modules["transformers"] = fake
    sys.modules["huggingface_hub"] = hub
    try:
        for ref, expect in ((str(tmp), []), ("some/repo", ["some/repo"])):
            called.clear()
            try:
                n._load_bundle(ref, "cpu", "sdpa")
            except OSError as e:
                assert "Repo id" in str(e), e
            else:
                raise AssertionError(f"{ref}: expected the OSError to survive")
            assert called == expect, (ref, called)
    finally:
        for k, v in saved.items():
            _restore(k, v)


def check_new_inputs_are_appended_not_inserted(n, tmp: Path) -> None:
    """ComfyUI restores widget values by POSITION. A new input in front of an
    existing one shifts every saved workflow by one -- the pack's own example
    workflow stores exactly [model_id, device, attention]."""
    spec = n.MOSSLoadModel.INPUT_TYPES()
    order = list(spec["required"]) + list(spec["optional"])
    assert order[:3] == ["model_id", "device", "attention"], order
    assert order.index("model_path") > order.index("attention"), order
    assert order.index("tokenizer_path") > order.index("model_path"), order

if __name__ == "__main__":
    sys.exit(main())
