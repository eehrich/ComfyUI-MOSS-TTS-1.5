"""Self-check for the generation and token plumbing that the 1.7B and the 8B do
differently.

The 8B (MossTTSDelay) writes its codes in a delay pattern and counts its
continuation prefix one row long; the 1.7B (MossTTSLocal) does neither. Every
check here pins a place where treating both builds alike was a real bug. The
fake processors carry exactly the attributes the pack asks for, and the delay
pattern is built the way MOSS builds it (channel c shifted down by c rows).

Plain asserts, no pytest (see test_model_paths.py for why). Run:

    python tests/test_generation.py
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
N_VQ = 4          # small, the mechanism does not depend on it
PAD = 1024


def _import_nodes():
    sys.modules.pop("folder_paths", None)
    spec = importlib.util.spec_from_file_location("moss_nodes_generation", ROOT / "nodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _delay(frames: torch.Tensor) -> torch.Tensor:
    """[T, n_vq] frames -> [T + n_vq - 1, n_vq] rows, channel c shifted by c."""
    t = frames.shape[0]
    rows = torch.full((t + N_VQ - 1, N_VQ), PAD, dtype=torch.long)
    for c in range(N_VQ):
        rows[c:c + t, c] = frames[:, c]
    return rows


def _de_delay(rows: torch.Tensor) -> torch.Tensor:
    t = rows.shape[0] - (N_VQ - 1)
    return torch.stack([rows[c:c + t, c] for c in range(N_VQ)], dim=1)


class _Config:
    n_vq = N_VQ
    audio_pad_code = PAD
    sampling_rate = 24000


class _DelayProcessor:
    """What the pack asks of the 8B processor."""
    model_config = _Config()
    apply_de_delay_pattern = staticmethod(_de_delay)

    def __init__(self):
        self.decoded_start_lengths: list[int] = []

    def decode(self, outputs):
        self.decoded_start_lengths = [int(sl) for sl, _ids in outputs]
        return [types.SimpleNamespace(audio_codes_list=[torch.zeros(1920)])]

    def decode_audio_codes(self, audio_tokens_list):  # the 8B signature: no return_stereo
        return [torch.zeros(int(t.shape[0]) * 1920) for t in audio_tokens_list]


class _LocalProcessor:
    """What the pack asks of the 1.7B processor: no delay pattern."""
    model_config = types.SimpleNamespace(n_vq=N_VQ, audio_pad_token_id=PAD,
                                         audio_pad_code=PAD, sampling_rate=48000)

    def __init__(self):
        self.decoded_start_lengths: list[int] = []
        self.return_stereo = None

    def decode(self, outputs):
        self.decoded_start_lengths = [int(sl) for sl, _ids in outputs]
        return [types.SimpleNamespace(audio_codes_list=[torch.zeros(2, 3840)])]

    def decode_audio_codes(self, audio_tokens_list, *, return_stereo=True):
        self.return_stereo = return_stereo
        return [torch.zeros(2, int(t.shape[0]) * 3840) for t in audio_tokens_list]


def _continuation_output(prefix: int, new: int):
    """An 8B continuation result: the audio_start marker row (all pad) in front of
    ONE delayed segment that holds the prefix frames and then the new ones.
    start_length is what MOSS reports: seq_len - (last im_start + 3), i.e. the
    marker plus the prefix rows."""
    frames = torch.randint(0, PAD, (prefix + new, N_VQ))
    marker = torch.full((1, N_VQ + 1), PAD, dtype=torch.long)
    marker[0, 0] = 7  # its text token
    body = torch.cat([torch.zeros(prefix + new + N_VQ - 1, 1, dtype=torch.long),
                      _delay(frames)], dim=1)
    return [(torch.tensor(1 + prefix), torch.cat([marker, body]))], frames


# --- 8B continuation: every new frame, none of the prefix ---------------------

def check_8b_continuation_tokens_keep_every_new_frame(n) -> None:
    outputs, frames = _continuation_output(prefix=40, new=25)
    tokens = n._extract_generated_codes(_DelayProcessor(), outputs)
    # MOSS's start_length counts the marker row that the segment no longer
    # holds; cut by it and the first new frame is lost -- 80 ms per seam.
    assert torch.equal(tokens, frames[40:]), (tokens.shape, "expected 25 new frames")


def check_8b_continuation_audio_is_trimmed_by_prefix_frames(n) -> None:
    outputs, _frames = _continuation_output(prefix=40, new=25)
    processor = _DelayProcessor()
    n._extract_audio(processor, outputs)
    assert processor.decoded_start_lengths == [40], processor.decoded_start_lengths


def check_8b_without_a_prefix_stays_at_zero(n) -> None:
    # Speak / Voice Clone: the marker is generated, not prompted -> 0, not -1.
    outputs, _frames = _continuation_output(prefix=0, new=10)
    outputs = [(torch.tensor(0), outputs[0][1])]
    processor = _DelayProcessor()
    n._extract_audio(processor, outputs)
    assert processor.decoded_start_lengths == [0], processor.decoded_start_lengths


def check_1_7b_start_length_is_left_alone(n) -> None:
    ids = torch.randint(0, PAD, (30, N_VQ + 1))
    processor = _LocalProcessor()
    n._extract_audio(processor, [(torch.tensor(12), ids)])
    assert processor.decoded_start_lengths == [12], processor.decoded_start_lengths


# --- Decode Tokens: the 8B has no return_stereo ------------------------------

def check_decode_tokens_runs_on_the_8b(n) -> None:
    tokens = torch.randint(0, PAD, (10, N_VQ))
    audio, frames = n.MOSSDecodeTokens().decode(
        {"processor": _DelayProcessor(), "sample_rate": 24000}, tokens, return_stereo=True)
    assert frames == 10
    assert audio["sample_rate"] == 24000 and audio["waveform"].shape[1] == 1, audio["waveform"].shape


def check_decode_tokens_honours_return_stereo_on_the_1_7b(n) -> None:
    tokens = torch.randint(0, PAD, (10, N_VQ))
    processor = _LocalProcessor()
    n.MOSSDecodeTokens().decode({"processor": processor, "sample_rate": 48000}, tokens,
                                return_stereo=False)
    assert processor.return_stereo is False, processor.return_stereo


def _model_node_calls(n, bundle: dict, text: str):
    """(name, call) for every node that takes a MOSS_MODEL, with inputs that pass
    its own checks; numbers are never reached, the model load comes first."""
    tokens = torch.zeros(10, N_VQ, dtype=torch.long)
    values = {"moss_model": bundle, "text": text, "previous_text": "hi", "language": "English",
              "instruction": "", "reference_tokens": tokens, "prev_tokens": tokens,
              "tokens": tokens, "audio": {"waveform": torch.zeros(1, 1, 480), "sample_rate": 48000}}
    for cls in (n.MOSSSpeak, n.MOSSVoiceClone, n.MOSSVoiceContinue,
                n.MOSSEncodeTokens, n.MOSSDecodeTokens):
        fn = getattr(cls(), cls.FUNCTION)
        params = inspect.signature(fn).parameters
        kwargs = {k: values[k] if k in values else 1 for k, p in params.items()
                  if k in values or p.default is inspect.Parameter.empty}
        yield cls.__name__, "text" in params, (lambda fn=fn, kw=kwargs: fn(**kw))


def check_every_model_node_loads_a_released_model_again(n) -> None:
    """Loading another model empties the old bundle; ComfyUI may still hand it
    to a node from its cache. But only after the node's own input checks: an
    empty text must not cost a model load."""
    class _Reloaded(Exception):
        pass

    def reload(*args):
        raise _Reloaded(args)

    released = {"device": "cpu", "sample_rate": 24000, "load_args": ("m", "cpu", "sdpa", None)}
    saved = n._load_bundle
    n._load_bundle = reload
    try:
        for name, _, call in _model_node_calls(n, released, "hi"):
            try:
                call()
            except _Reloaded as e:
                assert e.args[0] == ("m", "cpu", "sdpa", None), (name, e.args)
            else:
                raise AssertionError(f"{name} ran on a released bundle")
        for name, has_text, call in _model_node_calls(n, released, "  "):
            if has_text:
                try:
                    call()
                except ValueError:
                    continue
                raise AssertionError(f"{name} accepted an empty text")
    finally:
        n._load_bundle = saved


# --- the length cap is in frames on both builds -------------------------------

def check_overshoot_cap_counts_frames_on_both_builds(n) -> None:
    frames, prefix = 150, 40
    local = n._apply_overshoot_cap(4096, 100, 50, n._generation_row_overhead(_LocalProcessor()))
    assert local == frames, local
    # The 8B generates every row of its delayed layout that is not in the prompt,
    # then audio_end and im_end. A fresh prompt stops before audio_start ...
    delay = n._apply_overshoot_cap(4096, 100, 50, n._generation_row_overhead(_DelayProcessor()))
    assert delay == 1 + _delay(torch.zeros(frames, N_VQ)).shape[0] + 2, delay
    # ... a continuation's prompt already holds audio_start and the prefix rows.
    cont = n._apply_overshoot_cap(
        4096, 100, 50, n._generation_row_overhead(_DelayProcessor(), continuation=True))
    assert cont == _delay(torch.zeros(prefix + frames, N_VQ)).shape[0] - prefix + 2, cont
    # auto-EOS mode: max_new_tokens is frames too
    assert n._apply_overshoot_cap(300, None, 50, N_VQ + 2) == 300 + N_VQ + 2


# --- Load Tokens: the safe unpickler or nothing -------------------------------

def check_load_tokens_never_falls_back_to_the_full_unpickler(n) -> None:
    calls: list[object] = []
    real_load = torch.load

    def refusing_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        raise TypeError("the safe unpickler rejected this file")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "crafted.moss_tokens.pt"
        path.write_bytes(b"not really a tensor file")
        torch.load = refusing_load
        try:
            n.MOSSLoadTokens().load("", str(path))
        except TypeError:
            pass
        else:
            raise AssertionError("a rejected file was loaded anyway")
        finally:
            torch.load = real_load
    assert calls == [True], f"torch.load calls (weights_only): {calls}"


def check_network_paths_are_refused(n) -> None:
    if os.name != "nt":  # //x is a local path there
        try:
            n._resolve_tokens_path("//tmp/none.moss_tokens.pt")
        except FileNotFoundError:
            return
        raise AssertionError("//tmp/... was refused outside Windows")
    resolvers = {"token file": n._resolve_tokens_path,
                 "model_path": lambda p: n._resolve_model_ref("x", p),
                 "tokenizer_path": lambda p: n._resolve_tokenizer_ref("m", p)}
    for raw in (r"\\invalid.invalid\share\x.moss_tokens.pt",
                "//invalid.invalid/share/x.moss_tokens.pt",
                r"\/invalid.invalid/share/x.moss_tokens.pt",   # Windows: UNC as well
                r"/\invalid.invalid\share\x.moss_tokens.pt",
                r"\??\UNC\invalid.invalid\share\x.moss_tokens.pt"):  # NT spelling
        for what, resolve in resolvers.items():
            try:
                resolve(raw)
            except ValueError as e:
                assert "network" in str(e), e
            else:
                raise AssertionError(f"{what}: {raw!r} was accepted")
    # the dropdown's output/ prefix: base / "//host/share" IS //host/share
    for raw in ("output///invalid.invalid/share/x.moss_tokens.pt",
                r"output/\\invalid.invalid\share\x.moss_tokens.pt"):
        try:
            n._resolve_tokens_path(raw)
        except ValueError as e:
            assert "network" in str(e), e
        else:
            raise AssertionError(f"{raw!r} was accepted")


def check_a_token_file_on_comfyuis_own_share_is_allowed(n) -> None:
    """ComfyUI's output dir may be a share itself, and Save Tokens hands out a
    path inside it -- Load Tokens must take that back."""
    if os.name != "nt":
        return
    share = r"\\invalid.invalid\comfy_out"
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: share
    fp.get_input_directory = lambda: r"C:\moss_no_such_input"
    saved_fp = sys.modules.get("folder_paths")
    saved_is_file = Path.is_file
    sys.modules["folder_paths"] = fp
    Path.is_file = lambda self, *a, **kw: False  # never touch the share
    try:
        try:
            n._resolve_tokens_path(share + r"\moss_tokens\x.moss_tokens.pt")
        except FileNotFoundError:
            pass
        try:
            n._resolve_tokens_path(r"\\invalid.invalid\other\x.moss_tokens.pt")
        except ValueError as e:
            assert "network" in str(e), e
        else:
            raise AssertionError("a share next to the output dir was accepted")
    finally:
        Path.is_file = saved_is_file
        if saved_fp is None:
            sys.modules.pop("folder_paths", None)
        else:
            sys.modules["folder_paths"] = saved_fp


def check_token_values_must_be_codes(n) -> None:
    for bad in (torch.tensor([[-1, 0, 0, 0]]), torch.tensor([[3.9, 0.0, 0.0, 0.0]])):
        try:
            n._as_token_tensor(bad, "tokens", N_VQ)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad.tolist()}")
    ok = n._as_token_tensor(torch.tensor([[3.0, 0.0, 1023.0, 5.0]]), "tokens", N_VQ)
    assert ok.dtype == torch.long and ok.tolist() == [[3, 0, 1023, 5]], ok


# --- smaller ones ----------------------------------------------------------------

def check_estimate_counts_cjk_characters_and_words(n) -> None:
    est = n.MOSSEstimateTokens()
    english = "The quick brown fox jumps over the lazy dog near the river bank today"
    with_name = english.replace("fox", "fox 田中")
    base = est.estimate(english, 150)[0]
    assert est.estimate(with_name, 150)[0] <= base + 10, (base, est.estimate(with_name, 150))
    japanese = "今日は川のそばで素早い茶色の狐が怠け者の犬を飛び越えました"
    chars = sum(1 for ch in japanese if not ch.isspace())
    assert est.estimate(japanese, 150)[0] == est.estimate("x " * chars, 150)[0]
    # Chinese without spaces around Latin words: 15 characters + 4 words, not one
    # "word" (0.4 s) and not every Latin letter as a character.
    mixed = "我们在2024年发布了GPT-4o模型和Claude的对比测试。"
    assert est.estimate(mixed, 150)[0] == est.estimate("x " * 19, 150)[0], \
        est.estimate(mixed, 150)


def check_previous_audio_with_a_batch_concatenates(n) -> None:
    previous = {"waveform": torch.zeros(3, 2, 4800), "sample_rate": 48000}
    new = {"waveform": torch.zeros(1, 2, 3840), "sample_rate": 48000}
    full, frames = n._concat_full_audio(previous, new, 1, 1, 48000)
    assert tuple(full["waveform"].shape) == (1, 2, 8640), full["waveform"].shape


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
