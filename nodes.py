"""ComfyUI nodes for OpenMOSS MOSS-TTS v1.5.

Supports both variants (selected in the Load Model dropdown): the 1.7B
Local-Transformer (48 kHz, default) and the 8B full MOSS-TTS (24 kHz).

Ten nodes:
  - MOSSLoadModel:        loads the processor + model once, caches by (model_id, device).
  - MOSSSpeak:            MOSS_MODEL + text (no reference) -> AUDIO in a MOSS default voice.
  - MOSSVoiceClone:       MOSS_MODEL + reference AUDIO or MOSS_TOKENS + text -> cloned AUDIO out.
  - MOSSVoiceContinue:    MOSS_MODEL + previous AUDIO or MOSS_TOKENS + follow-up text -> continuation.
  - MOSSEstimateTokens:   text -> target_tokens estimate for duration control.
  - MOSSEncodeTokens:     AUDIO -> MOSS_TOKENS. Encode a voice reference ONCE.
  - MOSSDecodeTokens:     MOSS_TOKENS -> AUDIO. Audition/inspect codes without generating.
  - MOSSConcatTokens:     MOSS_TOKENS x2..4 -> MOSS_TOKENS. Build a sliding-window reference.
  - MOSSSaveTokens:       MOSS_TOKENS -> file in ComfyUI's output dir (+ path as STRING).
  - MOSSLoadTokens:       token file (dropdown over input/output dir, or an explicit
                          path) -> MOSS_TOKENS.

ComfyUI AUDIO shape: {"waveform": Tensor[B, C, T], "sample_rate": int}. Every AUDIO input
is encoded to MOSS codes IN MEMORY via the processor's tensor API (_comfy_audio_to_codes);
generated audio is wrapped back into an AUDIO dict on output. Nothing goes through a file,
so the pack never touches torchaudio's I/O backends (see _comfy_audio_to_codes).

MOSS_TOKENS is this pack's own type: a torch.LongTensor of shape [T, n_vq] holding raw MOSS
audio codes at 12.5 frames/s, i.e. one row = 80 ms of audio. The processor accepts such a
tensor everywhere it accepts a WAV path (see _resolve_audio_items in
processing_moss_tts.py: a torch.Tensor is taken as codes verbatim), and generate() EMITS
codes in exactly that layout. So a reference can be encoded once and reused forever --
which removes the codec encode from every request. That encode is the part of a cloned
request that does not parallelize: with a 70-90 s reference window it can dominate the
call (~24 s at concurrency 8 on a 5090) while generation itself scales fine.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torchaudio

logger = logging.getLogger("MOSS-TTS-ComfyUI")

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
# Dropdown labels: HF repo id + size in parentheses. Size suffix is stripped
# via _repo_id_from_label() before the HF `from_pretrained` call.
AVAILABLE_MODELS = [
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 (1.7B)",
    "OpenMOSS-Team/MOSS-TTS-v1.5 (8B)",
]
DEFAULT_MODEL_LABEL = AVAILABLE_MODELS[0]


def _repo_id_from_label(label: str) -> str:
    return label.split(" (", 1)[0].strip()
DEFAULT_LANGUAGES = (
    "Arabic", "Cantonese", "Chinese", "Czech", "Danish", "Dutch", "English",
    "Finnish", "French", "German", "Greek", "Hebrew", "Hindi", "Hungarian",
    "Italian", "Japanese", "Korean", "Macedonian", "Malay", "Persian (Farsi)",
    "Polish", "Portuguese", "Romanian", "Russian", "Spanish", "Swahili",
    "Swedish", "Tagalog", "Thai", "Turkish", "Vietnamese",
)
MOSS_FRAMES_PER_SECOND = 12.5
# Native sample rate of the loaded model. 1.7B Local-Transformer uses 48000 Hz,
# 8B MOSS-TTS uses 24000 Hz. Read from processor.model_config at load time and
# stored in each bundle -- all output audio dicts carry the actual rate so
# downstream ComfyUI nodes play back at the right speed.
MOSS_DEFAULT_SAMPLE_RATE = 48000  # only used as fallback reference in tooltips

# On-disk format for MOSS_TOKENS (MOSSSaveTokens / MOSSLoadTokens): a torch.save'd
# dict of plain tensors/ints/floats, so it loads back with weights_only=True (no
# pickle code execution -- important for a file type that travels between machines).
MOSS_TOKENS_SUFFIX = ".moss_tokens.pt"
MOSS_TOKENS_FORMAT_VERSION = 1
# MOSSLoadTokens' file dropdown lists BOTH ComfyUI directories. Entries from the
# output dir (where Save Tokens writes) carry this prefix, so identical basenames
# in input/ and output/ stay distinguishable; _resolve_tokens_path strips it again.
MOSS_TOKENS_OUTPUT_PREFIX = "output/"
# Shown when neither directory holds a token file yet: ComfyUI cannot render a
# COMBO input without entries, so the list is never empty. Selecting this label
# raises in _resolve_tokens_path instead of failing with a confusing "not found".
MOSS_TOKENS_NONE_LABEL = "(no token files found)"

# Shared "optional" input definition for the audio repetition penalty.
# MOSS's generate() natively supports audio_repetition_penalty (applied via
# _apply_repetition_penalty on the audio-token logits before sampling).
# 1.0 = neutral/off. Optional section so existing saved workflows keep
# validating without the field.
_REP_PENALTY_INPUT = (
    "FLOAT",
    {
        "default": 1.0,
        "min": 1.0,
        "max": 2.0,
        "step": 0.01,
        "tooltip": (
            "Penalty on recently generated audio tokens (1.0 = off). "
            "Mild values (1.05-1.15) suppress the classic autoregressive "
            "TTS failure modes -- droning, tempo freeze, smeared or "
            "looping syllables -- while leaving normal prosody untouched "
            "(it only bites on pathological repeats). Values above ~1.3 "
            "can distort legitimately repeated sounds ('nein, nein, nein')."
        ),
    },
)

# Shared "optional" inputs for the TEXT-stream sampler. MOSS v1.5 is a
# dual-stream (text + audio) model: generate() samples the text tokens that
# drive alignment/pacing with these, independently of the audio_* samplers
# that shape the acoustic codebook. Defaults match MOSS's own generate()
# defaults (1.0 / 1.0 / 50), so leaving them untouched changes nothing.
# Lowering text_temperature / tightening text_top_p|k stabilizes pacing and
# alignment without flattening the acoustic dynamics (which come from
# audio_temperature). Optional so existing saved workflows keep validating.
_TEXT_TEMPERATURE_INPUT = (
    "FLOAT",
    {
        "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
        "tooltip": (
            "Temperature for the TEXT stream (alignment/pacing), NOT the "
            "acoustics. MOSS default 1.0. Lower = steadier pacing/alignment; "
            "does not flatten the voice (that's audio_temperature)."
        ),
    },
)
_TEXT_TOP_P_INPUT = (
    "FLOAT",
    {
        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
        "tooltip": "Nucleus (top-p) cutoff for the TEXT stream. MOSS default 1.0 (off).",
    },
)
_TEXT_TOP_K_INPUT = (
    "INT",
    {
        "default": 50, "min": 1, "max": 200, "step": 1,
        "tooltip": "Top-k cutoff for the TEXT stream. MOSS default 50.",
    },
)

_MODEL_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}

# Attention backends offered in the loader dropdown. "auto" is the safe
# default (flash_attention_2 only if flash_attn is installed, else sdpa).
ATTENTION_CHOICES = ["auto", "sdpa", "flash_attention_2", "eager"]


def _resolve_dtype(device: str) -> tuple[torch.dtype, str]:
    """bf16 on CUDA (MOSS's training precision), fp32 on CPU (bf16 CPU kernels are patchy)."""
    if device.startswith("cuda"):
        return torch.bfloat16, "bfloat16"
    return torch.float32, "float32"


def _flash_attn_available(device: str) -> bool:
    """True only if the flash_attn package is importable and we're on CUDA."""
    if not device.startswith("cuda"):
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_attention(device: str, requested: str) -> str:
    """Resolve attn_implementation, robust on installs WITHOUT flash_attn.

    MOSS's remote code defaults to ``flash_attention_2``, which hard-crashes
    with ImportError if the ``flash_attn`` package is not installed — the
    common case on a fresh ComfyUI. So:
      * ``auto`` (default) uses flash_attention_2 only when flash_attn is
        actually importable on CUDA, otherwise ``sdpa`` (built into PyTorch,
        no extra install, fast on modern GPUs).
      * an explicit ``flash_attention_2`` request that can't be satisfied
        falls back to ``sdpa`` with a warning instead of crashing.
      * ``sdpa`` / ``eager`` are passed through.
    """
    req = (requested or "auto").lower()
    if req == "auto":
        return "flash_attention_2" if _flash_attn_available(device) else "sdpa"
    if req == "flash_attention_2" and not _flash_attn_available(device):
        logger.warning(
            "[MOSS-TTS] attn_implementation='flash_attention_2' requested but "
            "flash_attn is not installed (or device is CPU); falling back to "
            "'sdpa'. Install flash-attn to enable it."
        )
        return "sdpa"
    return req


def _load_bundle(model_id: str, device: str, attention: str = "auto") -> dict[str, Any]:
    attn = _resolve_attention(device, attention)
    key = (model_id, device, attn)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    dtype, dtype_name = _resolve_dtype(device)

    logger.info(f"[MOSS-TTS] loading processor '{model_id}' ...")
    from transformers import AutoModel, AutoProcessor
    # What the loaders are actually given. Stays the repo id unless the Windows
    # workaround below has to swap in a resolved local snapshot path.
    load_id: str = model_id
    try:
        processor = AutoProcessor.from_pretrained(load_id, trust_remote_code=True)
    except OSError as e:
        # Windows only, and only for some model builds: MOSS's own
        # processing_moss_tts.py does
        #     pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        # before handing the value to AutoConfig. On Windows that turns the repo
        # id "Org/Model" into "Org\Model", and the Hub rejects backslashes:
        #     Repo id must use alphanumeric chars, '-', '_' or '.'
        # MOSS-TTS-v1.5 (8B) has that line, MOSS-TTS-Local-Transformer-v1.5
        # does not — which is why only the 8B fails. Nothing we can fix in
        # their file (it lives in the HF module cache and is re-downloaded on
        # every model update), so sidestep it: resolve the repo to a local
        # directory first. Path() on a real directory is harmless, and
        # from_pretrained accepts a path just as well as a repo id.
        if "Repo id must use alphanumeric chars" not in str(e):
            raise
        from huggingface_hub import snapshot_download
        logger.info(
            "[MOSS-TTS] windows repo-id workaround: resolving "
            f"'{model_id}' to a local snapshot path"
        )
        load_id = snapshot_download(model_id)
        processor = AutoProcessor.from_pretrained(load_id, trust_remote_code=True)
    except AttributeError as e:
        # Newer MOSS model builds reference
        # processing_utils.MODALITY_TO_BASE_CLASS_MAPPING, which was introduced
        # in transformers 5.0.0 (it was AUTO_TO_BASE_CLASS_MAPPING in 4.x). Only
        # fire on the ACTUAL error — older cached model code loads fine on 4.x,
        # so we must not gate up-front. Turn the cryptic crash into a clear fix.
        if "MODALITY_TO_BASE_CLASS_MAPPING" in str(e):
            import sys
            import transformers
            tv = getattr(transformers, "__version__", "?")
            py = f"{sys.version_info.major}.{sys.version_info.minor}"
            if sys.version_info < (3, 10):
                raise RuntimeError(
                    f"This MOSS-TTS v1.5 model build needs transformers >= 5.0, "
                    f"but transformers 5.x requires Python >= 3.10 and your "
                    f"ComfyUI runs Python {py} (with transformers {tv}). Use a "
                    f"Python 3.10+ ComfyUI, or pin an older MOSS model build that "
                    f"runs on transformers 4.x."
                ) from e
            raise RuntimeError(
                f"This MOSS-TTS v1.5 model build needs transformers >= 5.0 (it "
                f"uses processing_utils.MODALITY_TO_BASE_CLASS_MAPPING, added in "
                f"5.0.0); you have transformers {tv} on Python {py}. Upgrade in "
                f"your ComfyUI Python environment:\n"
                f"    python -m pip install -U 'transformers>=5.0'"
            ) from e
        raise
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    logger.info(
        f"[MOSS-TTS] loading model '{model_id}' "
        f"(dtype={dtype_name}, device={device}, attn={attn}) ..."
    )
    model = AutoModel.from_pretrained(
        load_id, trust_remote_code=True, dtype=dtype,
        attn_implementation=attn,
    ).to(device)
    model.eval()

    sample_rate = int(getattr(processor.model_config, "sampling_rate", MOSS_DEFAULT_SAMPLE_RATE))
    logger.info(f"[MOSS-TTS] '{model_id}' native sample_rate={sample_rate} Hz")

    bundle = {
        "processor": processor,
        "model": model,
        "device": device,
        "dtype": dtype,
        "sample_rate": sample_rate,
    }
    _MODEL_CACHE[key] = bundle
    return bundle


class MOSSLoadModel:
    """Load and cache the MOSS-TTS processor + model.

    The bundle is memoised by (model_id, device) so subsequent workflow
    runs reuse the already-loaded weights with zero overhead. dtype is
    resolved internally (bfloat16 on CUDA, float32 on CPU).
    """

    DESCRIPTION = (
        "Loads a MOSS-TTS v1.5 processor + model. Two variants selectable: "
        "MOSS-TTS-Local-Transformer-v1.5 (~1.7B, MossTTSLocal architecture, "
        "our default, ~12 GB VRAM in bf16) or MOSS-TTS-v1.5 (~8B, "
        "MossTTSDelay architecture, ~22 GB VRAM, potentially better quality). "
        "First execution downloads weights (~9 GB / ~16 GB respectively) into "
        "the Hugging Face cache and moves them to the selected device. "
        "Subsequent runs reuse the cached bundle -> no re-load penalty. dtype "
        "is picked automatically: bfloat16 on CUDA, float32 on CPU."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model_id": (
                    AVAILABLE_MODELS,
                    {
                        "default": DEFAULT_MODEL_LABEL,
                        "tooltip": (
                            "Which MOSS model to load. Both are v1.5, same API, "
                            "31 languages, 48 kHz stereo, same 'tokens' / "
                            "duration semantics. Local-Transformer (~1.7B) is "
                            "smaller/faster (~12 GB VRAM), MOSS-TTS-v1.5 (~8B) "
                            "is the deeper MossTTSDelay model (~22 GB VRAM), "
                            "potentially better prosody/expressiveness. Fits "
                            "on RTX 5090 and 3090 both."
                        ),
                    },
                ),
                "device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "Where the model runs. CPU works but is very slow "
                            "(~50x slower than CUDA). Falls back to CPU "
                            "automatically when CUDA is unavailable."
                        ),
                    },
                ),
            },
            "optional": {
                "attention": (
                    ATTENTION_CHOICES,
                    {
                        "default": "auto",
                        "tooltip": (
                            "Attention backend. MOSS's model code defaults to "
                            "flash_attention_2, which CRASHES if the flash_attn "
                            "package isn't installed. 'auto' (recommended) uses "
                            "flash_attention_2 only when flash_attn is actually "
                            "available, otherwise 'sdpa' (built into PyTorch, no "
                            "extra install, fast). Force 'sdpa'/'eager' for "
                            "maximum compatibility, or 'flash_attention_2' only "
                            "if you installed flash-attn."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MOSS_MODEL",)
    RETURN_NAMES = ("moss_model",)
    OUTPUT_TOOLTIPS = ("Model bundle. Feed into any MOSS-TTS Speak / Voice Clone / Voice Continue node.",)
    FUNCTION = "load"
    CATEGORY = "MOSS TTS 1.5"

    def load(self, model_id: str, device: str, attention: str = "auto"):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("[MOSS-TTS] CUDA requested but not available; falling back to cpu.")
            device = "cpu"
        repo_id = _repo_id_from_label(model_id)
        bundle = _load_bundle(repo_id, device, attention=attention)
        return (bundle,)


def _seed(device: str, seed: int) -> None:
    torch.manual_seed(int(seed))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))


def _to_comfy_audio(audio_tensor: torch.Tensor, sample_rate: int) -> tuple[dict[str, Any], int, float]:
    """Return (AUDIO dict, tokens_generated, seconds) at the model's native sample rate."""
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    waveform = audio_tensor.detach().cpu().unsqueeze(0)  # -> [1, C, T]
    seconds = waveform.shape[-1] / float(sample_rate)
    tokens_generated = int(round(seconds * MOSS_FRAMES_PER_SECOND))
    return ({"waveform": waveform, "sample_rate": int(sample_rate)}, tokens_generated, seconds)


def _sanitize_target_tokens(target_tokens: int) -> int | None:
    if target_tokens and target_tokens > 0:
        return int(target_tokens)
    return None


def _require_text(text: str, field: str = "text") -> str:
    """Return text.strip(), raising on empty/whitespace-only input.

    MOSS's generate() has no guard against an empty prompt: with nothing to
    say it never emits the EOS token and keeps sampling audio frames until it
    hits max_new_tokens — minutes of garbage on a hang that looks like the
    node is frozen. Fail fast with a clear message instead.
    """
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError(
            f"MOSS-TTS: '{field}' is empty. Provide non-empty text to speak — "
            "an empty prompt makes MOSS generate audio until max_new_tokens "
            "(it never stops on its own), which looks like a hang."
        )
    return stripped


def _apply_overshoot_cap(max_new_tokens: int, target_tokens: int | None, overshoot: int) -> int:
    """Cap effective max_new_tokens when target_tokens is set, to prevent MOSS runaway.

    If target_tokens is unset (None), returns max_new_tokens unchanged (auto-EOS mode).
    Otherwise returns min(max_new_tokens, target_tokens + overshoot).
    """
    if target_tokens is None:
        return int(max_new_tokens)
    return min(int(max_new_tokens), int(target_tokens) + max(0, int(overshoot)))


def _to_stereo_at(waveform: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    """Return waveform as [1, 2, T] at target_sr. Handles mono + rate mismatch."""
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [C, T] -> [1, C, T]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # [T] -> [1, 1, T]

    waveform = waveform.detach().cpu().to(torch.float32)
    if int(source_sr) != int(target_sr):
        waveform = torchaudio.functional.resample(waveform, int(source_sr), int(target_sr))

    channels = int(waveform.shape[1])
    if channels == 1:
        waveform = waveform.repeat(1, 2, 1)  # mono -> stereo
    elif channels > 2:
        waveform = waveform[:, :2, :]
    return waveform


def _concat_full_audio(
    previous_audio: dict[str, Any],
    new_audio_dict: dict[str, Any],
    prefix_frames: int,
    new_tokens: int,
    target_sr: int,
) -> tuple[dict[str, Any], int]:
    """Concatenate previous_audio + new segment as a fresh ComfyUI AUDIO dict at target_sr."""
    prev_wave = _to_stereo_at(previous_audio["waveform"], int(previous_audio["sample_rate"]), target_sr)
    new_wave = new_audio_dict["waveform"].to(torch.float32)  # already at target_sr from _to_comfy_audio
    if new_wave.shape[1] == 1:
        new_wave = new_wave.repeat(1, 2, 1)
    full_wave = torch.cat([prev_wave, new_wave], dim=-1)
    return ({"waveform": full_wave, "sample_rate": int(target_sr)}, int(prefix_frames + new_tokens))


def _extract_audio(processor: Any, outputs: Any) -> torch.Tensor:
    """Decode + pull the first audio tensor, with a clear error if MOSS returned nothing."""
    decoded = processor.decode(outputs)
    if not decoded or decoded[0] is None:
        raise RuntimeError(
            "MOSS returned no decodable audio (empty content in generation). "
            "Try a different seed, longer text, or check the input for illegal chars."
        )
    codes = decoded[0].audio_codes_list
    if not codes:
        raise RuntimeError("MOSS returned an assistant message with empty audio_codes_list.")
    return codes[0]


def _processor_n_vq(processor: Any) -> int:
    """RVQ depth (number of audio codebooks) of the loaded model.

    Never hardcode this: the processor validates every code tensor against
    model_config.n_vq (_assert_fixed_nq) and raises if it disagrees, and the two
    MOSS v1.5 builds are free to differ.
    """
    config = getattr(processor, "model_config", None)
    n_vq = getattr(config, "n_vq", None)
    if n_vq is None:
        raise RuntimeError(
            "MOSS-TTS: processor.model_config.n_vq is unavailable -- cannot determine "
            "the RVQ depth. Reload the model with MOSS-TTS Load Model."
        )
    return int(n_vq)


def _audio_pad_token_id(processor: Any) -> int:
    """Code value that marks a NON-audio row in the unified [T, n_vq + 1] stream."""
    config = getattr(processor, "model_config", None)
    pad_id = getattr(config, "audio_pad_token_id", None)
    if pad_id is None:
        raise RuntimeError(
            "MOSS-TTS: processor.model_config.audio_pad_token_id is unavailable -- "
            "cannot separate audio frames from prompt rows."
        )
    return int(pad_id)


def _as_token_tensor(value: Any, field: str, n_vq: int | None = None) -> torch.Tensor:
    """Validate + normalise a MOSS_TOKENS value to a contiguous CPU LongTensor [T, n_vq]."""
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"MOSS-TTS: '{field}' must be a MOSS_TOKENS tensor, got {type(value).__name__}. "
            "Wire it from MOSS-TTS Encode Tokens / Concat Tokens / Load Tokens or from a "
            "Voice Clone / Voice Continue 'tokens' output."
        )
    if value.dim() != 2:
        raise ValueError(
            f"MOSS-TTS: '{field}' must have shape [frames, n_vq], got {tuple(value.shape)}."
        )
    if n_vq is not None and int(value.shape[1]) != int(n_vq):
        raise ValueError(
            f"MOSS-TTS: '{field}' has n_vq={int(value.shape[1])} but the loaded model uses "
            f"n_vq={int(n_vq)}. Tokens are model-specific -- re-encode the reference with "
            "MOSS-TTS Encode Tokens using the model you are generating with."
        )
    return value.detach().to(dtype=torch.long).cpu().contiguous()


def _token_seconds(frames: int) -> float:
    """Frames -> seconds at MOSS's fixed 12.5 fps (1 frame = 80 ms)."""
    return int(frames) / MOSS_FRAMES_PER_SECOND


def _comfy_audio_to_codes(
    processor: Any, audio: dict[str, Any], field: str = "audio"
) -> tuple[torch.Tensor, float]:
    """Encode a ComfyUI AUDIO dict to MOSS codes [T, n_vq] in memory. Returns (codes, seconds).

    The ONE audio -> codes path of this pack (Encode Tokens plus the legacy
    reference_audio / previous_audio inputs). It goes through the processor's
    TENSOR api, never through a file, and that is deliberate: the file api
    (encode_audios_from_path) calls torchaudio.load(), and torchaudio 2.9+
    dropped its own I/O backends and delegates save/load to torchcodec. In a
    venv without torchcodec (or soundfile) -- which is what ComfyUI ships --
    BOTH torchaudio.save and torchaudio.load raise ImportError, so writing a
    temp WAV and letting the processor read it back broke the whole audio
    input path. encode_audios_from_wav only uses torchaudio.functional.resample
    (a pure tensor op), so it works on any torchaudio build.

    The processor wants a [C, T] float32 waveform plus its sample rate: it
    duplicates mono to stereo, keeps the first two channels of anything wider,
    resamples to the model's native rate and loudness-normalises -- exactly
    what it used to do with the samples it read back from the WAV. The only
    difference to that round trip is that the codes are no longer computed from
    16-bit PCM: a dropped quantisation step, not a change of meaning, so the
    codes are not bit-identical to the old path but carry the same voice.
    """
    n_vq = _processor_n_vq(processor)
    waveform: torch.Tensor = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.dim() == 3:
        waveform = waveform[0]  # ComfyUI AUDIO is [B, C, T] -- take the first batch
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [T] -> [1, T]
    waveform = waveform.detach().cpu().to(torch.float32)
    seconds = waveform.shape[-1] / float(sample_rate)

    with torch.inference_mode():
        encoded = processor.encode_audios_from_wav([waveform], sample_rate, n_vq=n_vq)
    if not encoded:
        raise RuntimeError(
            f"MOSS's audio tokenizer returned no codes for '{field}'. "
            "Check that the audio is non-empty and not pure silence."
        )
    # .clone() leaves inference-mode territory, so the tensor can be freely
    # concatenated / saved / cached by ComfyUI afterwards.
    return _as_token_tensor(encoded[0].clone(), f"encoded {field}", n_vq), seconds


def _extract_generated_codes(processor: Any, outputs: Any) -> torch.Tensor:
    """Pull the NEWLY generated audio codes out of a raw model.generate() result.

    Code-domain twin of :func:`_extract_audio`: same segmentation rules as the
    processor's ``_parse_audio_codes``, but it stops short of the vocoder and
    returns the [T, n_vq] LongTensor MOSS just emitted.

    ``generate()`` returns ``[(start_length, generation_ids), ...]`` with
    ``generation_ids`` shaped ``[T, n_vq + 1]``: channel 0 is the text stream,
    channels ``1:`` are the audio codebooks. Rows that hold audio_pad in ALL
    codebooks are prompt/text rows; what remains are one or more CONTIGUOUS
    audio segments. ``start_length`` counts the prompt rows after the last
    audio-start marker:
      * generation / voice clone -> 0. The reference codes sit before that
        marker, so they are not in ``generation_ids`` at all (and if a build
        ever does emit them as a leading segment, the start_length rule below
        drops it, exactly like _parse_audio_codes does).
      * continuation -> the length of the prompt audio, which shares ONE
        segment with the new audio. Unlike the audio path (which can only trim
        by sample proportion after vocoding) we slice ``[start_length:]``, an
        exact frame-accurate cut.
    """
    items = list(outputs or [])
    if not items:
        raise RuntimeError(
            "MOSS returned no generation output at all (empty result from generate())."
        )
    start_length, generation_ids = items[0]
    if not isinstance(generation_ids, torch.Tensor) or generation_ids.dim() != 2:
        raise RuntimeError(
            "MOSS returned an unexpected generation payload; expected a [T, n_vq + 1] "
            f"tensor, got {type(generation_ids).__name__}."
        )

    n_vq = _processor_n_vq(processor)
    if int(generation_ids.shape[1]) != n_vq + 1:
        raise RuntimeError(
            f"MOSS returned {int(generation_ids.shape[1])} channels, expected n_vq + 1 = "
            f"{n_vq + 1}. Model and processor disagree on the RVQ depth."
        )

    audio_codes = generation_ids[:, 1:].to(dtype=torch.long).cpu()
    is_pad = audio_codes.eq(_audio_pad_token_id(processor)).all(dim=1)
    non_pad = ~is_pad
    if not bool(non_pad.any().item()):
        raise RuntimeError(
            "MOSS emitted no audio frames (generation is all padding). "
            "Try a different seed, longer text, or check the input for illegal chars."
        )

    idx = torch.nonzero(non_pad).squeeze(1)
    breaks = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
    segment_indices = [idx] if breaks.numel() == 0 else list(torch.tensor_split(idx, breaks.cpu().tolist()))
    code_segments = [audio_codes[segment] for segment in segment_indices]

    start_length = int(start_length)
    if start_length > 0 and code_segments:
        first_length = int(code_segments[0].shape[0])
        if start_length >= first_length:
            # The whole first segment was prompt audio -- same call as
            # _parse_audio_codes' trim_ratio >= 1.0 branch.
            code_segments = code_segments[1:]
        else:
            code_segments[0] = code_segments[0][start_length:]

    if not code_segments or int(code_segments[0].shape[0]) == 0:
        raise RuntimeError(
            "MOSS returned only prompt audio and no new frames. With a target_tokens "
            "hint this usually means the budget was already spent by the prefix; "
            "otherwise try a different seed or longer text."
        )
    return code_segments[0].contiguous()


def _build_tokens_payload(tokens: torch.Tensor) -> dict[str, Any]:
    """Serialisable dict for MOSSSaveTokens (plain tensors/ints/floats only)."""
    return {
        "format": "moss_tokens",
        "format_version": MOSS_TOKENS_FORMAT_VERSION,
        "audio_codes": tokens,
        "frames": int(tokens.shape[0]),
        "n_vq": int(tokens.shape[1]),
        "frames_per_second": float(MOSS_FRAMES_PER_SECOND),
    }


def _tokens_from_payload(payload: Any, source: str) -> torch.Tensor:
    """Inverse of :func:`_build_tokens_payload`; also accepts a bare tensor."""
    if isinstance(payload, torch.Tensor):
        return _as_token_tensor(payload, f"tokens from {source}")
    if not isinstance(payload, dict) or "audio_codes" not in payload:
        raise ValueError(
            f"MOSS-TTS: '{source}' is not a MOSS token file (expected a dict with an "
            "'audio_codes' entry, written by MOSS-TTS Save Tokens)."
        )
    version = int(payload.get("format_version", MOSS_TOKENS_FORMAT_VERSION))
    if version > MOSS_TOKENS_FORMAT_VERSION:
        raise ValueError(
            f"MOSS-TTS: '{source}' was written with token format v{version}, this build "
            f"understands up to v{MOSS_TOKENS_FORMAT_VERSION}. Update the node pack."
        )
    return _as_token_tensor(payload["audio_codes"], f"tokens from {source}")


def _folder_paths() -> Any:
    """ComfyUI's folder_paths module, imported lazily so nodes.py stays importable."""
    try:
        import folder_paths  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - only outside ComfyUI
        raise RuntimeError(
            "MOSS-TTS: ComfyUI's folder_paths module is unavailable -- the Save/Load "
            "Tokens nodes only work inside a running ComfyUI."
        ) from e
    return folder_paths


def _comfy_dir(kind: str) -> Path:
    """ComfyUI's output ('output') or input directory (anything else)."""
    folder_paths = _folder_paths()
    if kind == "output":
        return Path(folder_paths.get_output_directory())
    return Path(folder_paths.get_input_directory())


def _list_token_files() -> list[str]:
    """Token files in ComfyUI's input and output dir, exactly as the loader takes them.

    Recursive, because Save Tokens writes into a subfolder by default
    ('output/moss_tokens/...') and _resolve_tokens_path accepts subpaths -- so the
    dropdown never offers something the loader cannot open, and never hides
    something it could. Output-dir entries get the MOSS_TOKENS_OUTPUT_PREFIX.
    Never raises: outside ComfyUI, or with an unreadable directory, the list is
    simply empty (INPUT_TYPES runs on every /object_info request).
    """
    entries: set[str] = set()
    for kind in ("input", "output"):
        prefix = MOSS_TOKENS_OUTPUT_PREFIX if kind == "output" else ""
        try:
            base = _comfy_dir(kind).resolve()
            for file in base.rglob("*" + MOSS_TOKENS_SUFFIX):
                if file.is_file():
                    entries.add(prefix + file.relative_to(base).as_posix())
        except Exception as e:  # noqa: BLE001 - a broken dir must never break the node list
            logger.debug(f"[MOSS-TTS] token file scan skipped {kind} dir: {type(e).__name__}: {e}")
    return sorted(entries, key=lambda name: (name.lower(), name))


def _selected_tokens_path(path: str, path_override: str = "") -> str:
    """'path_override' wins when it is non-empty, otherwise the dropdown selection."""
    return (path_override or "").strip() or (path or "")


def _resolve_tokens_path(path: str) -> Path:
    """Resolve a token filename against ComfyUI's input dir, then its output dir.

    Three shapes are accepted. An entry from the Load Tokens dropdown (relative,
    carrying MOSS_TOKENS_OUTPUT_PREFIX when it came from the output dir, which is
    tried there first). Any other relative path, resolved input dir first and
    output dir second -- unchanged, so a value stored in an older workflow still
    resolves exactly as it did. And an absolute path, used as-is, so the STRING
    path returned by MOSS-TTS Save Tokens can be fed straight back in on a later
    run (an HTTP-driven pipeline saves in one prompt and loads in the next).
    Relative paths must stay inside the input/output directory.
    """
    raw = (path or "").strip().strip('"')
    if not raw:
        raise ValueError(
            "MOSS-TTS: no token file selected. Pick one from the 'path' dropdown, or put a "
            "filename inside ComfyUI's input dir (e.g. "
            "'moss_tokens/base_voice_00001_.moss_tokens.pt') or an absolute path into "
            "'path_override'."
        )
    if raw == MOSS_TOKENS_NONE_LABEL:
        raise RuntimeError(
            "MOSS-TTS: no token file exists yet, so the 'path' dropdown is empty. Run "
            "MOSS-TTS Save Tokens once (it writes into ComfyUI's output directory) and "
            "reload the page, or put a path into 'path_override'."
        )
    candidate = Path(raw)
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"MOSS-TTS: token file not found: {candidate}")

    # (directory, relative path) attempts. A prefixed dropdown entry is tried in the
    # output dir first; the unprefixed input->output order stays as a fallback, so
    # nothing that resolved before this node grew a dropdown stops resolving now.
    attempts: list[tuple[str, str]] = []
    normalised = raw.replace("\\", "/")
    if normalised.lower().startswith(MOSS_TOKENS_OUTPUT_PREFIX):
        stripped = normalised[len(MOSS_TOKENS_OUTPUT_PREFIX):]
        if stripped:
            attempts.append(("output", stripped))
    attempts += [("input", raw), ("output", raw)]

    tried: list[Path] = []
    for kind, relative in attempts:
        base = _comfy_dir(kind).resolve()
        resolved = (base / relative).resolve()
        if base not in resolved.parents:
            raise ValueError(
                f"MOSS-TTS: '{raw}' escapes ComfyUI's {kind} directory. Use a path inside "
                "it, or an absolute path."
            )
        if resolved.is_file():
            return resolved
        tried.append(resolved)
    raise FileNotFoundError(
        "MOSS-TTS: token file not found. Tried: " + ", ".join(str(p) for p in tried)
    )


# Shared "optional" input definitions for the MOSS_TOKENS path on the generation
# nodes. Both live in the optional section: ComfyUI cannot express "required
# unless another input is wired", so the node checks at runtime and raises when
# neither the audio nor the tokens input is connected. Note that MOSS_TOKENS and
# AUDIO are link-only inputs (no widget), so adding/moving them does NOT shift
# the positional widgets_values array of saved workflows.
_REFERENCE_TOKENS_INPUT = (
    "MOSS_TOKENS",
    {
        "tooltip": (
            "Pre-encoded voice reference (MOSS audio codes). When wired this "
            "REPLACES 'reference_audio' and skips the codec encode completely -- "
            "the expensive, non-parallelising part of every cloned request. "
            "Encode the base voice ONCE with MOSS-TTS Encode Tokens (or reuse the "
            "'tokens' output of a previous generation) and keep feeding the same "
            "tensor. Tokens are model-specific: n_vq must match the loaded model. "
            "Same length rule as reference_audio: give MOSS at least ~10 s of "
            "reference (>=125 frames). Around 5 s is NOT enough acoustic evidence "
            "and the output comes out as gibberish, not as a weaker clone."
        ),
    },
)
_PREV_TOKENS_INPUT = (
    "MOSS_TOKENS",
    {
        "tooltip": (
            "Audio codes of the previous segment (NOT the 'previous_tokens' frame "
            "COUNT below). When wired this REPLACES 'previous_audio': no codec "
            "re-encode, and the prefix length is taken exactly from "
            "the tensor. Feed the 'tokens' output of the preceding Voice Clone / "
            "Voice Continue node. RULE: 'previous_text' must transcribe EXACTLY "
            "what these codes contain -- shorten the token stream and you must "
            "shorten the transcript to the same point. A mismatched pair produces "
            "gibberish, not a slightly-off voice."
        ),
    },
)
_TOKENS_OUTPUT_TOOLTIP = (
    "Audio codes MOSS just generated, shape [frames, n_vq] at 12.5 fps (1 row = "
    "80 ms). Feed into the next node's reference_tokens / prev_tokens (optionally "
    "through MOSS-TTS Concat Tokens) to keep the whole chain encode-free. These are "
    "the RAW emitted codes: they cover the untrimmed segment, so they include the "
    "frames removed by head_trim_frames from the AUDIO output."
)


class MOSSSpeak:
    """Text-to-speech without a voice reference (MOSS's built-in 'None' voice path)."""

    DESCRIPTION = (
        "Generates speech without a reference audio. MOSS was trained on a "
        "'None' placeholder path (see processing_moss_tts.py: else-branch of "
        "_build_generation_or_voice_clone_codes) that lets it pick a voice "
        "based on language + the 'instruction' hint. Since there is no audio "
        "reference here, 'instruction' is the ONLY voice-steering knob -- "
        "worth trying things like 'male, warm, elderly narrator' or 'young "
        "female, cheerful, energetic'. The 'tokens' output carries the codes "
        "MOSS just emitted, so a freshly invented voice can go straight into "
        "Voice Clone / Voice Continue without ever being re-encoded."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "moss_model": (
                    "MOSS_MODEL",
                    {"tooltip": "Model bundle produced by MOSS-TTS Load Model."},
                ),
                "text": (
                    "STRING",
                    {
                        "default": "Hello, this is a test.",
                        "multiline": True,
                        "tooltip": (
                            "Text to synthesize. For silence gaps use "
                            "punctuation (., --, ...) or chain a follow-up "
                            "run with an empty-audio spacer."
                        ),
                    },
                ),
                "language": (
                    list(DEFAULT_LANGUAGES),
                    {
                        "default": "English",
                        "tooltip": (
                            "Language hint. Also nudges MOSS toward a "
                            "language-typical base voice."
                        ),
                    },
                ),
                "instruction": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Describe the voice (male/female, age, tone, style)",
                        "tooltip": (
                            "Voice description passed to MOSS's 'instruction' "
                            "channel. Without a reference audio this is the "
                            "only steering knob for voice character. Examples: "
                            "'male, warm, elderly narrator', 'young female, "
                            "cheerful', 'deep voice, dramatic, slow'."
                        ),
                    },
                ),
                "audio_temperature": (
                    "FLOAT",
                    {
                        "default": 1.7,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": (
                            "Sampling temperature. MOSS default is 1.7. "
                            "Lower -> more deterministic and safer, higher -> "
                            "more expressive but noisier."
                        ),
                    },
                ),
                "audio_top_p": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus (top-p) sampling cutoff.",
                    },
                ),
                "audio_top_k": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Top-k sampling cutoff.",
                    },
                ),
                "target_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 65536,
                        "step": 1,
                        "tooltip": (
                            "Optional target duration hint, in audio frames. "
                            "0 = disabled (model decides via EOS). At 12.5 "
                            "frames/s: 375 tokens ~30 s, 750 ~60 s. Chain a "
                            "MOSS-TTS Estimate Tokens node to compute this "
                            "from the text."
                        ),
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 65536,
                        "step": 128,
                        "tooltip": (
                            "Safety cap on generated audio frames. MOSS runs "
                            "at 12.5 frames/s, so the default 4096 caps output "
                            "at ~5 min. The model stops on its own EOS token, "
                            "so real output is usually much shorter."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "tooltip": "Random seed. Same seed + same inputs -> identical output.",
                    },
                ),
                "target_overshoot_frames": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 65536,
                        "step": 10,
                        "tooltip": (
                            "Runaway safety cap: when target_tokens > 0, "
                            "MOSS may only exceed it by this many frames. "
                            "Effective max_new_tokens = min(max_new_tokens, "
                            "target_tokens + target_overshoot_frames). "
                            "Default 50 = 4 s slack at 12.5 fps. Prevents the "
                            "5.5-min hang MOSS occasionally does with "
                            "pathologically short text. Ignored when "
                            "target_tokens = 0 (auto-EOS mode)."
                        ),
                    },
                ),
            },
            "optional": {
                "audio_repetition_penalty": _REP_PENALTY_INPUT,
                "text_temperature": _TEXT_TEMPERATURE_INPUT,
                "text_top_p": _TEXT_TOP_P_INPUT,
                "text_top_k": _TEXT_TOP_K_INPUT,
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "MOSS_TOKENS")
    RETURN_NAMES = ("audio", "tokens_generated", "tokens")
    OUTPUT_TOOLTIPS = (
        "Generated audio at 48 kHz stereo, ready for SaveAudio / PreviewAudio.",
        "Number of audio frames MOSS actually generated (frames, not samples). "
        "At 12.5 fps this equals duration_seconds * 12.5.",
        _TOKENS_OUTPUT_TOOLTIP,
    )
    FUNCTION = "generate"
    CATEGORY = "MOSS TTS 1.5"

    def generate(
        self,
        moss_model: dict[str, Any],
        text: str,
        language: str,
        instruction: str,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        target_tokens: int,
        max_new_tokens: int,
        seed: int,
        target_overshoot_frames: int = 50,
        audio_repetition_penalty: float = 1.0,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
    ) -> tuple[dict[str, Any], int, torch.Tensor]:
        processor = moss_model["processor"]
        model = moss_model["model"]
        device = moss_model["device"]
        _seed(device, seed)

        clean_text = _require_text(text, "text")
        build_kwargs: dict[str, Any] = {
            "text": clean_text,
            "language": language,
        }
        if instruction.strip():
            build_kwargs["instruction"] = instruction.strip()
        tok_hint = _sanitize_target_tokens(target_tokens)
        if tok_hint is not None:
            build_kwargs["tokens"] = tok_hint

        effective_max = _apply_overshoot_cap(max_new_tokens, tok_hint, target_overshoot_frames)
        logger.info(
            f"[MOSS-TTS] speak text_chars={len(build_kwargs['text'])} "
            f"lang={language} instruction={'set' if instruction.strip() else 'none'} "
            f"target_tokens={tok_hint or 'auto'} effective_max={effective_max} "
            f"(user_max={max_new_tokens}, overshoot={target_overshoot_frames}) "
            f"temperature={audio_temperature} top_p={audio_top_p} top_k={audio_top_k} "
            f"rep_penalty={audio_repetition_penalty}"
        )
        conversation = [processor.build_user_message(**build_kwargs)]
        batch = processor([conversation], mode="generation")

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=effective_max,
                audio_temperature=float(audio_temperature),
                audio_top_p=float(audio_top_p),
                audio_top_k=int(audio_top_k),
                audio_repetition_penalty=float(audio_repetition_penalty),
                text_temperature=float(text_temperature),
                text_top_p=float(text_top_p),
                text_top_k=int(text_top_k),
            )

        audio_tensor: torch.Tensor = _extract_audio(processor, outputs)
        generated_tokens: torch.Tensor = _extract_generated_codes(processor, outputs)

        sample_rate = int(moss_model["sample_rate"])
        audio_dict, tokens_generated, seconds = _to_comfy_audio(audio_tensor, sample_rate)
        logger.info(
            f"[MOSS-TTS] speak done seconds={seconds:.2f} "
            f"tokens_generated={tokens_generated} emitted_frames={int(generated_tokens.shape[0])} "
            f"sample_rate={sample_rate}"
        )
        return (audio_dict, tokens_generated, generated_tokens)


class MOSSVoiceClone:
    """Generate speech in the cloned voice from a reference AUDIO."""

    DESCRIPTION = (
        "Zero-shot voice cloning. Feed a reference audio clip (any length, "
        "any language supported by MOSS) plus target text and MOSS returns "
        "the target text spoken in that voice at 48 kHz stereo. Give the "
        "reference at least ~10 s: below that (a ~5 s clip) MOSS has too "
        "little acoustic evidence and returns gibberish rather than a rough "
        "clone. MOSS does NOT accept a reference transcript here -- only the "
        "audio clip and an optional style hint via the 'instruction' input. "
        "Wire 'reference_tokens' instead of 'reference_audio' to reuse an "
        "already-encoded reference and skip the codec encode entirely."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "moss_model": (
                    "MOSS_MODEL",
                    {
                        "tooltip": "Model bundle produced by MOSS-TTS Load Model.",
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "default": "Hello, this is a test.",
                        "multiline": True,
                        "tooltip": (
                            "Text to synthesize in the cloned voice. For "
                            "silence gaps use punctuation (., --, ...) or "
                            "chain a second Voice Clone / Voice Continue run "
                            "with an empty-audio spacer between them."
                        ),
                    },
                ),
                "language": (
                    list(DEFAULT_LANGUAGES),
                    {
                        "default": "English",
                        "tooltip": (
                            "Explicit language hint. Setting this consistently "
                            "improves prosody and pronunciation vs. relying on "
                            "language detection from the text."
                        ),
                    },
                ),
                "instruction": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional style/direction hint (NOT a reference transcript)",
                        "tooltip": (
                            "Free-form style hint passed to MOSS's built-in "
                            "'instruction' channel, e.g. 'warm and slow', "
                            "'excited', 'whispered'. This is NOT a transcript "
                            "of the reference audio -- MOSS has no such input."
                        ),
                    },
                ),
                "audio_temperature": (
                    "FLOAT",
                    {
                        "default": 1.7,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": (
                            "Sampling temperature. MOSS default is 1.7. "
                            "Lower -> more deterministic and safer, higher -> "
                            "more expressive but noisier."
                        ),
                    },
                ),
                "audio_top_p": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus (top-p) sampling cutoff.",
                    },
                ),
                "audio_top_k": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Top-k sampling cutoff.",
                    },
                ),
                "target_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 65536,
                        "step": 1,
                        "tooltip": (
                            "Optional target duration hint, in audio frames. "
                            "0 = disabled (model decides via EOS). At 12.5 "
                            "frames/s: 375 tokens ~30 s, 750 ~60 s, 3750 ~5 min. "
                            "Chain a MOSS-TTS Estimate Tokens node to compute "
                            "this from the text."
                        ),
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 65536,
                        "step": 128,
                        "tooltip": (
                            "Safety cap on generated audio frames. MOSS runs "
                            "at 12.5 frames/s, so the default 4096 caps output "
                            "at ~5 min. The model stops on its own EOS token, "
                            "so real output is usually much shorter."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "tooltip": "Random seed. Same seed + same inputs -> identical output.",
                    },
                ),
                "target_overshoot_frames": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 65536,
                        "step": 10,
                        "tooltip": (
                            "Runaway safety cap: when target_tokens > 0, "
                            "MOSS may only exceed it by this many frames. "
                            "Effective max_new_tokens = min(max_new_tokens, "
                            "target_tokens + target_overshoot_frames). "
                            "Default 50 = 4 s slack at 12.5 fps. Prevents the "
                            "5.5-min hang MOSS occasionally does with "
                            "pathologically short text. Ignored when "
                            "target_tokens = 0 (auto-EOS mode)."
                        ),
                    },
                ),
            },
            "optional": {
                "reference_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Voice reference. Any ComfyUI AUDIO source works "
                            "(LoadAudio, another node's output, etc.). Give it at "
                            "least ~10 s, 10-20 s is the sweet spot; MOSS v1.5 also "
                            "handles long references reliably. TOO SHORT IS NOT A "
                            "WEAKER CLONE, IT IS GIBBERISH: around 5 s the model has "
                            "not enough acoustic evidence to lock onto the voice and "
                            "the output is unusable. Required unless "
                            "'reference_tokens' is wired -- it is only declared "
                            "optional because ComfyUI has no way to express "
                            "'required unless that other input is connected'. "
                            "Ignored when reference_tokens is wired."
                        ),
                    },
                ),
                "reference_tokens": _REFERENCE_TOKENS_INPUT,
                "audio_repetition_penalty": _REP_PENALTY_INPUT,
                "text_temperature": _TEXT_TEMPERATURE_INPUT,
                "text_top_p": _TEXT_TOP_P_INPUT,
                "text_top_k": _TEXT_TOP_K_INPUT,
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "MOSS_TOKENS")
    RETURN_NAMES = ("audio", "tokens_generated", "tokens")
    OUTPUT_TOOLTIPS = (
        "Generated audio at 48 kHz stereo, ready for SaveAudio / PreviewAudio.",
        "Number of audio frames MOSS actually generated (frames, not samples). "
        "At 12.5 fps this equals duration_seconds * 12.5.",
        _TOKENS_OUTPUT_TOOLTIP,
    )
    FUNCTION = "generate"
    CATEGORY = "MOSS TTS 1.5"

    def generate(
        self,
        moss_model: dict[str, Any],
        text: str,
        language: str,
        instruction: str,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        target_tokens: int,
        max_new_tokens: int,
        seed: int,
        reference_audio: dict[str, Any] | None = None,
        reference_tokens: torch.Tensor | None = None,
        target_overshoot_frames: int = 50,
        audio_repetition_penalty: float = 1.0,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
    ) -> tuple[dict[str, Any], int, torch.Tensor]:
        processor = moss_model["processor"]
        model = moss_model["model"]
        device = moss_model["device"]
        _seed(device, seed)

        if reference_tokens is None and reference_audio is None:
            raise ValueError(
                "MOSS-TTS Voice Clone: no voice reference. Wire either "
                "'reference_audio' (an AUDIO clip, encoded on every run) or "
                "'reference_tokens' (pre-encoded MOSS_TOKENS, no encode at all)."
            )

        clean_text = _require_text(text, "text")
        # Both branches end up as a [T, n_vq] code tensor: the processor's
        # _resolve_audio_items takes such a tensor verbatim. The token path just
        # skips the codec pass -- neither path touches the disk.
        if reference_tokens is not None:
            if reference_audio is not None:
                logger.info(
                    "[MOSS-TTS] clone reference_tokens wired -- ignoring reference_audio."
                )
            n_vq = _processor_n_vq(processor)
            reference_item: torch.Tensor = _as_token_tensor(reference_tokens, "reference_tokens", n_vq)
            ref_frames = int(reference_item.shape[0])
            ref_desc = f"tokens frames={ref_frames} seconds={_token_seconds(ref_frames):.2f}"
        else:
            reference_item, ref_seconds = _comfy_audio_to_codes(
                processor, reference_audio, "reference_audio"
            )
            ref_desc = (
                f"audio seconds={ref_seconds:.2f} -> frames={int(reference_item.shape[0])} "
                "(encoded on this run)"
            )

        build_kwargs: dict[str, Any] = {
            "text": clean_text,
            "reference": [reference_item],
            "language": language,
        }
        if instruction.strip():
            build_kwargs["instruction"] = instruction.strip()
        tok_hint = _sanitize_target_tokens(target_tokens)
        if tok_hint is not None:
            build_kwargs["tokens"] = tok_hint

        effective_max = _apply_overshoot_cap(max_new_tokens, tok_hint, target_overshoot_frames)
        logger.info(
            f"[MOSS-TTS] clone text_chars={len(build_kwargs['text'])} "
            f"lang={language} instruction={'set' if instruction.strip() else 'none'} "
            f"reference={ref_desc} "
            f"target_tokens={tok_hint or 'auto'} effective_max={effective_max} "
            f"(user_max={max_new_tokens}, overshoot={target_overshoot_frames}) "
            f"temperature={audio_temperature} top_p={audio_top_p} top_k={audio_top_k} "
            f"rep_penalty={audio_repetition_penalty}"
        )
        conversation = [processor.build_user_message(**build_kwargs)]
        batch = processor([conversation], mode="generation")

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=effective_max,
                audio_temperature=float(audio_temperature),
                audio_top_p=float(audio_top_p),
                audio_top_k=int(audio_top_k),
                audio_repetition_penalty=float(audio_repetition_penalty),
                text_temperature=float(text_temperature),
                text_top_p=float(text_top_p),
                text_top_k=int(text_top_k),
            )

        audio_tensor: torch.Tensor = _extract_audio(processor, outputs)
        generated_tokens: torch.Tensor = _extract_generated_codes(processor, outputs)

        sample_rate = int(moss_model["sample_rate"])
        audio_dict, tokens_generated, seconds = _to_comfy_audio(audio_tensor, sample_rate)
        logger.info(
            f"[MOSS-TTS] clone done seconds={seconds:.2f} "
            f"tokens_generated={tokens_generated} emitted_frames={int(generated_tokens.shape[0])} "
            f"sample_rate={sample_rate}"
        )
        return (audio_dict, tokens_generated, generated_tokens)


class MOSSVoiceContinue:
    """Continue an existing MOSS-TTS clip: same voice, more text."""

    DESCRIPTION = (
        "Extends a previously generated MOSS-TTS clip. Internally MOSS is a "
        "PREFIX-continuation model -- it needs the ORIGINAL text that "
        "produced 'previous_audio' so it can lock onto the exact point in "
        "the script where the audio left off, then produce audio for the "
        "follow-up 'text'. This node concatenates 'previous_text' + 'text' "
        "into the full script MOSS conditions on; the voice comes from the "
        "prior audio itself (no separate reference input needed). Because "
        "MOSS aligns the spoken prefix against that script, the pair has to "
        "be consistent: 'previous_text' must transcribe 'previous_audio' / "
        "'prev_tokens' exactly -- a mismatched pair (trimmed audio, untrimmed "
        "transcript; wrong paragraph pasted) yields gibberish. Output "
        "contains only the newly-generated audio (concatenate with the "
        "input if you want the full stream). Wire 'prev_tokens' instead of "
        "'previous_audio' to hand MOSS the prior segment's codes directly -- "
        "no codec re-encode, frame-exact prefix length."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "moss_model": (
                    "MOSS_MODEL",
                    {"tooltip": "Model bundle produced by MOSS-TTS Load Model."},
                ),
                "previous_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "The exact text that produced 'previous_audio'",
                        "tooltip": (
                            "The exact text that produced 'previous_audio' / "
                            "'prev_tokens' -- its transcript. MOSS aligns the "
                            "spoken reference against this text to find its 'where "
                            "am I in the script?' state, so it must match "
                            "word-for-word (punctuation matters). IF THE TEXT DOES "
                            "NOT TRANSCRIBE THE AUDIO, THE OUTPUT IS GARBAGE -- not "
                            "a degraded voice, gibberish. Trimmed the reference WAV? "
                            "Trim this text to the same point. Pasted the wrong "
                            "paragraph? Same result."
                        ),
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Follow-up text to speak next",
                        "tooltip": (
                            "New text to speak after the previous audio ends. "
                            "Internally concatenated as: previous_text + ' ' + "
                            "text -> full script. Empty is legal but MOSS "
                            "will then close out almost immediately -- supply "
                            "real follow-up text for meaningful output."
                        ),
                    },
                ),
                "language": (
                    list(DEFAULT_LANGUAGES),
                    {
                        "default": "English",
                        "tooltip": "Language hint for the follow-up text.",
                    },
                ),
                "audio_temperature": (
                    "FLOAT",
                    {"default": 1.7, "min": 0.1, "max": 3.0, "step": 0.05,
                     "tooltip": "Sampling temperature (MOSS default 1.7)."},
                ),
                "audio_top_p": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Nucleus (top-p) sampling cutoff."},
                ),
                "audio_top_k": (
                    "INT",
                    {"default": 25, "min": 1, "max": 200, "step": 1,
                     "tooltip": "Top-k sampling cutoff."},
                ),
                "target_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 65536,
                        "step": 1,
                        "tooltip": (
                            "Target length of the NEW continuation segment, in "
                            "audio frames (12.5 fps). 0 = disabled (model "
                            "decides via EOS). The node adds the measured "
                            "prefix length internally, because MOSS reads its "
                            "'tokens' hint as TOTAL (prefix + new) in "
                            "continuation mode. Chain a MOSS-TTS Estimate "
                            "Tokens node on the FOLLOW-UP text to compute this."
                        ),
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 65536,
                        "step": 128,
                        "tooltip": (
                            "Safety cap on newly-generated audio frames. "
                            "Same units as in Voice Clone: 12.5 fps -> 4096 "
                            "caps continuation at ~5 min."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 0xFFFFFFFF,
                     "tooltip": "Random seed. Same seed + same inputs -> identical output."},
                ),
                "previous_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 65536,
                        "step": 1,
                        "tooltip": (
                            "Exact frame count of 'previous_audio'. Wire the "
                            "'tokens_generated' output of the preceding "
                            "Speak / Voice Clone / Voice Continue node into "
                            "this input for a precise handoff. Leave at 0 to "
                            "measure from the audio duration (fine for "
                            "externally-loaded WAVs, off by <=1 frame due to "
                            "rounding). Ignored when 'prev_tokens' is wired: "
                            "the code tensor already carries the exact length."
                        ),
                    },
                ),
                "head_trim_frames": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": (
                            "Extra frames to trim from the START of the new "
                            "audio (1 frame = 80 ms at 12.5 fps). MOSS's "
                            "decoder trims the prefix by SAMPLE proportion, "
                            "and its conv-based 48 kHz codec has a receptive "
                            "field that spans frame boundaries -- so the "
                            "last prefix frame can bleed audibly into the "
                            "start of the returned continuation. Default 1 "
                            "(~80 ms) removes it in most cases. Set 0 to "
                            "disable, higher if the bleed is longer."
                        ),
                    },
                ),
                "target_overshoot_frames": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 65536,
                        "step": 10,
                        "tooltip": (
                            "Runaway safety cap on the NEW continuation "
                            "segment: when target_tokens > 0, MOSS may only "
                            "exceed target_tokens by this many frames. "
                            "Effective max_new_tokens = min(max_new_tokens, "
                            "target_tokens + target_overshoot_frames). "
                            "Default 50 = 4 s slack at 12.5 fps. Prevents "
                            "long hangs on pathological inputs. Ignored "
                            "when target_tokens = 0."
                        ),
                    },
                ),
            },
            "optional": {
                "previous_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Prior MOSS output to continue from. Typically the "
                            "AUDIO output of a preceding MOSS-TTS Voice Clone / "
                            "Voice Continue node. Must be paired with the exact "
                            "'previous_text' that produced it -- a clip and a "
                            "transcript that do not describe the same speech yield "
                            "gibberish, so cut both to the same point or neither. "
                            "Required unless "
                            "'prev_tokens' is wired -- it is only declared "
                            "optional because ComfyUI has no way to express "
                            "'required unless that other input is connected'. "
                            "Still worth wiring alongside prev_tokens if you "
                            "want the 'full_audio' output: without it there is "
                            "no prior waveform to prepend."
                        ),
                    },
                ),
                "prev_tokens": _PREV_TOKENS_INPUT,
                "audio_repetition_penalty": _REP_PENALTY_INPUT,
                "text_temperature": _TEXT_TEMPERATURE_INPUT,
                "text_top_p": _TEXT_TOP_P_INPUT,
                "text_top_k": _TEXT_TOP_K_INPUT,
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "AUDIO", "INT", "MOSS_TOKENS")
    RETURN_NAMES = ("audio", "tokens_generated", "full_audio", "full_tokens", "tokens")
    OUTPUT_TOOLTIPS = (
        "New segment only, head-trimmed. Use this for per-segment QC / "
        "preview -- you hear just the delta MOSS produced this call.",
        "Frames of the NEW segment only (frames, not samples). At 12.5 fps "
        "this equals duration_seconds * 12.5.",
        "Cumulative audio: previous_audio + new segment concatenated at "
        "48 kHz stereo. Wire this into the NEXT Continue's previous_audio "
        "when the same speaker keeps talking across segments. Falls back to "
        "the new segment alone when only prev_tokens (no previous_audio) is "
        "wired -- there is no prior waveform to prepend then.",
        "Cumulative frame count: prefix + new. Wire into the next Continue's "
        "previous_tokens for a precise handoff without re-measurement. Equals "
        "'tokens_generated' when no previous_audio is wired.",
        _TOKENS_OUTPUT_TOOLTIP,
    )
    FUNCTION = "generate"
    CATEGORY = "MOSS TTS 1.5"

    def generate(
        self,
        moss_model: dict[str, Any],
        previous_text: str,
        text: str,
        language: str,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        target_tokens: int,
        max_new_tokens: int,
        seed: int,
        previous_audio: dict[str, Any] | None = None,
        prev_tokens: torch.Tensor | None = None,
        previous_tokens: int = 0,
        head_trim_frames: int = 1,
        target_overshoot_frames: int = 50,
        audio_repetition_penalty: float = 1.0,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
    ) -> tuple[dict[str, Any], int, dict[str, Any], int, torch.Tensor]:
        processor = moss_model["processor"]
        model = moss_model["model"]
        device = moss_model["device"]
        _seed(device, seed)

        if prev_tokens is None and previous_audio is None:
            raise ValueError(
                "MOSS-TTS Voice Continue: nothing to continue from. Wire either "
                "'previous_audio' (an AUDIO clip, re-encoded on every run) or "
                "'prev_tokens' (the previous segment's MOSS_TOKENS, no encode at all)."
            )

        new = _require_text(text, "text")
        prev = (previous_text or "").strip()
        full_text = (prev + " " + new).strip() if prev else new

        # Both branches end up as a [T, n_vq] code tensor for the assistant message
        # (_resolve_audio_items takes such a tensor verbatim); the audio branch just
        # pays for the codec pass. prefix_frames keeps its old precedence
        # (wired > measured) so the frame accounting of saved workflows is unchanged --
        # the codec's own frame count only rides along in the log for comparison.
        if prev_tokens is not None:
            if previous_audio is not None:
                logger.info(
                    "[MOSS-TTS] continue prev_tokens wired -- using codes, "
                    "previous_audio only feeds the full_audio output."
                )
            n_vq = _processor_n_vq(processor)
            prior_item: torch.Tensor = _as_token_tensor(prev_tokens, "prev_tokens", n_vq)
            prefix_frames = int(prior_item.shape[0])
            prefix_source = "tokens"
        else:
            prior_item, prior_seconds_measured = _comfy_audio_to_codes(
                processor, previous_audio, "previous_audio"
            )
            encoded_frames = int(prior_item.shape[0])
            if previous_tokens and previous_tokens > 0:
                prefix_frames = int(previous_tokens)
                prefix_source = f"wired, encoded={encoded_frames}"
            else:
                prefix_frames = int(round(prior_seconds_measured * MOSS_FRAMES_PER_SECOND))
                prefix_source = f"measured, encoded={encoded_frames}"
        prior_seconds = (
            previous_audio["waveform"].shape[-1] / float(previous_audio["sample_rate"])
            if previous_audio is not None
            else _token_seconds(prefix_frames)
        )

        build_kwargs: dict[str, Any] = {
            "text": full_text,
            "language": language,
        }
        tok_hint = _sanitize_target_tokens(target_tokens)
        total_tokens = None
        if tok_hint is not None:
            # MOSS interprets `tokens` in continuation as TOTAL (prefix + new).
            # The node's `target_tokens` input is defined as "frames of NEW audio",
            # so we add the prefix here (exact with prev_tokens, measured otherwise).
            total_tokens = tok_hint + prefix_frames
            build_kwargs["tokens"] = total_tokens

        user_msg = processor.build_user_message(**build_kwargs)
        assistant_msg = processor.build_assistant_message(audio_codes_list=[prior_item])
        conversation = [user_msg, assistant_msg]

        # max_new_tokens in continuation mode caps NEW frames only (prefix is in input_ids).
        # target_tokens is defined as "new frames", so the overshoot cap applies directly.
        effective_max = _apply_overshoot_cap(max_new_tokens, tok_hint, target_overshoot_frames)

        logger.info(
            f"[MOSS-TTS] continue prev_chars={len(prev)} new_chars={len(new)} "
            f"full_chars={len(full_text)} lang={language} "
            f"target_new={tok_hint or 'auto'} effective_max={effective_max} "
            f"(user_max={max_new_tokens}, overshoot={target_overshoot_frames}) "
            f"prefix_frames={prefix_frames} ({prefix_source}) "
            f"total_tokens_to_moss={total_tokens or 'auto'} prior_seconds={prior_seconds:.2f} "
            f"rep_penalty={audio_repetition_penalty}"
        )
        batch = processor([conversation], mode="continuation")

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=effective_max,
                audio_temperature=float(audio_temperature),
                audio_top_p=float(audio_top_p),
                audio_top_k=int(audio_top_k),
                audio_repetition_penalty=float(audio_repetition_penalty),
                text_temperature=float(text_temperature),
                text_top_p=float(text_top_p),
                text_top_k=int(text_top_k),
            )

        audio_tensor: torch.Tensor = _extract_audio(processor, outputs)
        generated_tokens: torch.Tensor = _extract_generated_codes(processor, outputs)

        sample_rate = int(moss_model["sample_rate"])
        trim = max(0, int(head_trim_frames))
        if trim > 0 and audio_tensor.numel() > 0:
            # samples_per_frame = sample_rate / frames_per_second (48000/12.5=3840 for 1.7B, 24000/12.5=1920 for 8B)
            trim_samples = int(round(trim * sample_rate / MOSS_FRAMES_PER_SECOND))
            trim_samples = min(trim_samples, audio_tensor.shape[-1] - 1)
            if trim_samples > 0:
                audio_tensor = audio_tensor[..., trim_samples:]

        audio_dict, tokens_generated, seconds = _to_comfy_audio(audio_tensor, sample_rate)

        if previous_audio is not None:
            full_audio_dict, full_tokens = _concat_full_audio(
                previous_audio, audio_dict, prefix_frames, tokens_generated, sample_rate
            )
        else:
            # Token-only continuation: no prior waveform exists in this graph, so
            # "full" degrades to the new segment. Chain MOSS-TTS Concat Tokens on
            # the 'tokens' output if you want the cumulative CODE stream.
            full_audio_dict, full_tokens = audio_dict, tokens_generated

        logger.info(
            f"[MOSS-TTS] continue done seconds={seconds:.2f} "
            f"tokens_generated={tokens_generated} emitted_frames={int(generated_tokens.shape[0])} "
            f"head_trim_frames={trim} "
            f"full_seconds={full_audio_dict['waveform'].shape[-1]/sample_rate:.2f} "
            f"full_tokens={full_tokens} sample_rate={sample_rate}"
        )
        return (audio_dict, tokens_generated, full_audio_dict, full_tokens, generated_tokens)


class MOSSEncodeTokens:
    """Encode a ComfyUI AUDIO clip into MOSS audio codes, once."""

    DESCRIPTION = (
        "Turns an AUDIO clip into MOSS_TOKENS (raw audio codes) with the "
        "model's own codec. Use this for the base voice reference so it never "
        "has to be encoded again: Voice Clone re-encodes its reference_audio on "
        "EVERY run, and with a long reference that encode can dominate the "
        "request -- it is a single codec pass that does not parallelise across "
        "concurrent requests, unlike generation itself. Encode once, wire the "
        "MOSS_TOKENS into reference_tokens (and/or save it with MOSS-TTS Save "
        "Tokens), and every later run starts straight at generation. Output is "
        "[frames, n_vq] at 12.5 fps, so 1 frame = 80 ms. Tokens are tied to the "
        "loaded model -- re-encode when you switch between the 1.7B and 8B build."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "moss_model": (
                    "MOSS_MODEL",
                    {"tooltip": "Model bundle produced by MOSS-TTS Load Model."},
                ),
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Audio to encode. Resampled to the model's native "
                            "rate and loudness-normalised by the processor "
                            "exactly as the Voice Clone reference path does, so "
                            "the resulting codes are interchangeable with it. "
                            "Two rules the clip itself has to satisfy, both of "
                            "which produce GIBBERISH (not a weaker voice) when "
                            "broken: give it at least ~10 s -- around 5 s is not "
                            "enough acoustic evidence for MOSS to lock onto the "
                            "voice; and if these codes later serve as a Voice "
                            "Continue prefix, whatever text you pass as "
                            "'previous_text' must transcribe THIS clip exactly -- "
                            "trim the audio and you must trim the transcript to "
                            "the same point."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MOSS_TOKENS", "INT")
    RETURN_NAMES = ("tokens", "frames")
    OUTPUT_TOOLTIPS = (
        "Audio codes, shape [frames, n_vq]. Wire into Voice Clone "
        "'reference_tokens', Voice Continue 'prev_tokens', MOSS-TTS Concat "
        "Tokens or MOSS-TTS Save Tokens.",
        "Number of code frames. Divide by 12.5 for seconds.",
    )
    FUNCTION = "encode"
    CATEGORY = "MOSS TTS 1.5"

    def encode(self, moss_model: dict[str, Any], audio: dict[str, Any]) -> tuple[torch.Tensor, int]:
        processor = moss_model["processor"]
        tokens, seconds = _comfy_audio_to_codes(processor, audio, "audio")
        frames = int(tokens.shape[0])
        logger.info(
            f"[MOSS-TTS] encode_tokens seconds={seconds:.2f} "
            f"sample_rate={int(audio['sample_rate'])} "
            f"frames={frames} n_vq={int(tokens.shape[1])}"
        )
        return (tokens, frames)


class MOSSDecodeTokens:
    """Decode MOSS_TOKENS back into an audible ComfyUI AUDIO clip."""

    DESCRIPTION = (
        "Inverse of MOSS-TTS Encode Tokens: sends MOSS audio codes back through "
        "the model's own vocoder and returns a normal ComfyUI AUDIO dict. This "
        "completes the token API (encode / decode / concat / save / load) and is "
        "what makes a token stream auditable -- listen to what a saved token file "
        "actually holds, hear the exact reference window a MOSS-TTS Concat Tokens "
        "result builds, or check a 'tokens' output frame for frame, all WITHOUT "
        "generating anything. Especially useful before a long run: a reference "
        "that sounds wrong here will clone wrong. Decoding is a pure codec pass, "
        "no sampling -- deterministic, no seed. Codes are model-specific, so "
        "decode with the model that produced them (n_vq is validated)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "moss_model": (
                    "MOSS_MODEL",
                    {
                        "tooltip": (
                            "Model bundle produced by MOSS-TTS Load Model. The "
                            "vocoder is part of the model, so decode with the same "
                            "variant that encoded/emitted the codes -- it also "
                            "defines the output sample rate (48 kHz for the 1.7B "
                            "Local-Transformer, 24 kHz for the 8B)."
                        ),
                    },
                ),
                "tokens": (
                    "MOSS_TOKENS",
                    {
                        "tooltip": (
                            "Codes to turn back into audio, shape [frames, n_vq] at "
                            "12.5 fps (1 row = 80 ms). Any MOSS_TOKENS source works: "
                            "MOSS-TTS Encode / Concat / Load Tokens, or the 'tokens' "
                            "output of Speak / Voice Clone / Voice Continue."
                        ),
                    },
                ),
            },
            "optional": {
                "return_stereo": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Channel layout, passed straight to the processor's "
                            "decode_audio_codes(return_stereo=...). True (default) "
                            "keeps the codec's native stereo -- identical to what "
                            "every generate node in this pack outputs, so a decoded "
                            "'tokens' output lines up with its own 'audio'. False "
                            "averages the codec channels into one mono channel: "
                            "smaller, but no longer bit-identical to the generate "
                            "nodes' audio."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "frames")
    OUTPUT_TOOLTIPS = (
        "Decoded audio at the model's native sample rate (stereo unless "
        "return_stereo is off), ready for PreviewAudio / SaveAudio.",
        "Number of code frames decoded. Divide by 12.5 for seconds.",
    )
    FUNCTION = "decode"
    CATEGORY = "MOSS TTS 1.5"

    def decode(
        self,
        moss_model: dict[str, Any],
        tokens: torch.Tensor,
        return_stereo: bool = True,
    ) -> tuple[dict[str, Any], int]:
        processor = moss_model["processor"]
        n_vq = _processor_n_vq(processor)
        tensor = _as_token_tensor(tokens, "tokens", n_vq)
        frames = int(tensor.shape[0])
        if frames == 0:
            raise ValueError(
                "MOSS-TTS Decode Tokens: 'tokens' holds 0 frames -- there is nothing "
                "to decode. Check the upstream Encode / Concat / Load Tokens node."
            )

        # processor.decode_audio_codes takes exactly the [T, n_vq] layout this pack
        # passes around (it transposes to [n_vq, T] itself) and returns ONE waveform
        # per code segment: [C, T] float32 on CPU, or [T] mono with return_stereo=False.
        # It is the same call _parse_audio_codes makes behind processor.decode(), minus
        # the prompt-row segmentation and the start_length trim -- so decoding a
        # 'tokens' output reproduces the AUDIO output it was emitted with.
        with torch.inference_mode():
            decoded = processor.decode_audio_codes([tensor], return_stereo=bool(return_stereo))
        if not decoded or decoded[0] is None:
            raise RuntimeError(
                "MOSS's vocoder returned no audio for these codes. Check that the "
                "tokens come from this model and are not empty."
            )
        # .clone() leaves inference-mode territory, so ComfyUI may cache / serialise
        # the waveform afterwards (same reason as in _comfy_audio_to_codes).
        waveform: torch.Tensor = decoded[0].clone()

        sample_rate = int(moss_model["sample_rate"])
        # measured_frames comes from the decoded duration; it must land on `frames`
        # (the codec is a fixed 12.5 fps). Logged rather than returned, so a codec
        # that ever disagrees is visible instead of silently reshaping the count.
        audio_dict, measured_frames, seconds = _to_comfy_audio(waveform, sample_rate)
        logger.info(
            f"[MOSS-TTS] decode_tokens frames={frames} seconds={seconds:.2f} "
            f"n_vq={n_vq} return_stereo={bool(return_stereo)} "
            f"channels={int(audio_dict['waveform'].shape[1])} "
            f"measured_frames={measured_frames} sample_rate={sample_rate}"
        )
        return (audio_dict, frames)


class MOSSConcatTokens:
    """Concatenate MOSS_TOKENS streams along the time axis."""

    DESCRIPTION = (
        "Joins two to four MOSS_TOKENS streams end to end. Built for "
        "sliding-window references: keep a fixed base anchor (the voice you "
        "cloned from) and append the tokens of the most recent segment(s), then "
        "feed the result into Voice Clone's 'reference_tokens'. That gives the "
        "same continuity as concatenating reference WAVs, without touching "
        "audio at all -- no decode, no re-encode, no resample. All inputs must "
        "come from the same model (identical n_vq)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "tokens_a": (
                    "MOSS_TOKENS",
                    {"tooltip": "First stream. For a sliding window this is the base voice anchor."},
                ),
                "tokens_b": (
                    "MOSS_TOKENS",
                    {"tooltip": "Second stream, appended after tokens_a (e.g. the most recent segment)."},
                ),
            },
            "optional": {
                "tokens_c": ("MOSS_TOKENS", {"tooltip": "Optional third stream, appended after tokens_b."}),
                "tokens_d": ("MOSS_TOKENS", {"tooltip": "Optional fourth stream, appended after tokens_c."}),
            },
        }

    RETURN_TYPES = ("MOSS_TOKENS", "INT")
    RETURN_NAMES = ("tokens", "frames")
    OUTPUT_TOOLTIPS = (
        "Concatenated codes, shape [sum(frames), n_vq].",
        "Total frame count. Divide by 12.5 for seconds -- handy to keep a "
        "sliding-window reference inside a duration budget.",
    )
    FUNCTION = "concat"
    CATEGORY = "MOSS TTS 1.5"

    def concat(
        self,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
        tokens_c: torch.Tensor | None = None,
        tokens_d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        parts: list[torch.Tensor] = []
        n_vq: int | None = None
        for field, value in (
            ("tokens_a", tokens_a),
            ("tokens_b", tokens_b),
            ("tokens_c", tokens_c),
            ("tokens_d", tokens_d),
        ):
            if value is None:
                continue
            tensor = _as_token_tensor(value, field, n_vq)
            if n_vq is None:
                n_vq = int(tensor.shape[1])
            parts.append(tensor)

        merged = torch.cat(parts, dim=0)
        frames = int(merged.shape[0])
        logger.info(
            f"[MOSS-TTS] concat_tokens parts={[int(p.shape[0]) for p in parts]} "
            f"frames={frames} seconds={_token_seconds(frames):.2f} n_vq={n_vq}"
        )
        return (merged, frames)


class MOSSSaveTokens:
    """Persist MOSS_TOKENS to ComfyUI's output directory."""

    DESCRIPTION = (
        "Writes MOSS_TOKENS to a file in ComfyUI's output directory and returns "
        "the absolute path. Format: torch.save of a dict "
        "{format, format_version, audio_codes [frames, n_vq], frames, n_vq, "
        "frames_per_second}, saved with a '" + MOSS_TOKENS_SUFFIX + "' suffix and "
        "loadable with weights_only=True (plain tensors only -- no pickled code). "
        "Use it to encode a voice reference once and reuse it across runs and "
        "across processes: an HTTP-driven pipeline saves the base voice one time, "
        "then every later prompt just points MOSS-TTS Load Tokens at the file. "
        "The path is also reported in the node's UI output, so it can be read "
        "back from ComfyUI's /history response."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "tokens": (
                    "MOSS_TOKENS",
                    {"tooltip": "Codes to persist, e.g. from MOSS-TTS Encode Tokens or a 'tokens' output."},
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "moss_tokens/voice",
                        "tooltip": (
                            "Path prefix inside ComfyUI's output directory. "
                            "Subfolders are created automatically and a counter "
                            "plus '" + MOSS_TOKENS_SUFFIX + "' is appended, e.g. "
                            "'moss_tokens/voice' -> "
                            "'output/moss_tokens/voice_00001_" + MOSS_TOKENS_SUFFIX + "'."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_TOOLTIPS = (
        "Absolute path of the written file. Feed it into MOSS-TTS Load Tokens "
        "(in this or a later run) or read it from ComfyUI's /history output.",
    )
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "MOSS TTS 1.5"

    def save(self, tokens: torch.Tensor, filename_prefix: str) -> dict[str, Any]:
        tensor = _as_token_tensor(tokens, "tokens")
        # get_save_image_path is ComfyUI's shared "resolve prefix + counter" helper
        # (SaveImage / SaveAudio use it too): it creates subfolders and blocks
        # prefixes that would escape the output directory.
        full_output_folder, filename, counter, _subfolder, _prefix = _folder_paths().get_save_image_path(
            filename_prefix, str(_comfy_dir("output"))
        )
        path = Path(full_output_folder) / f"{filename}_{counter:05}_{MOSS_TOKENS_SUFFIX}"
        torch.save(_build_tokens_payload(tensor), str(path))

        frames = int(tensor.shape[0])
        logger.info(
            f"[MOSS-TTS] save_tokens frames={frames} seconds={_token_seconds(frames):.2f} "
            f"n_vq={int(tensor.shape[1])} path={path}"
        )
        return {"ui": {"text": [str(path)]}, "result": (str(path),)}


class MOSSLoadTokens:
    """Load MOSS_TOKENS written by MOSS-TTS Save Tokens."""

    DESCRIPTION = (
        "Reads a token file written by MOSS-TTS Save Tokens back into "
        "MOSS_TOKENS. 'path' is a dropdown of the token files found in "
        "ComfyUI's input and output directory (output entries prefixed "
        "'output/'); 'path_override' beats the dropdown whenever it is not "
        "empty and takes any relative or absolute path -- a dropdown cannot "
        "accept a link, that STRING can, so the 'path' output of Save Tokens "
        "can be wired straight in. Loaded with weights_only=True, i.e. tensors "
        "only, no code execution. This is the piece that makes an encode-once "
        "pipeline work across runs and process restarts: the base voice is "
        "encoded a single time, ever."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        # A list-typed input renders as a dropdown. Keep 'path' in first position
        # and 'path_override' last: widgets_values in a saved workflow is a
        # positional array, so STRING -> COMBO keeps the stored value in its slot
        # and the appended widget just falls back to its default in old workflows.
        return {
            "required": {
                "path": (
                    _list_token_files() or [MOSS_TOKENS_NONE_LABEL],
                    {
                        "tooltip": (
                            "Token file to load. Scanned recursively from "
                            "ComfyUI's input and output directory; entries from "
                            "the output directory -- where Save Tokens writes -- "
                            "carry an 'output/' prefix. Files written after the "
                            "page was loaded appear on reload. A path stored by "
                            "an older workflow that is not in this list still "
                            "resolves the way it always did."
                        ),
                    },
                ),
            },
            "optional": {
                "path_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "moss_tokens/voice_00001_" + MOSS_TOKENS_SUFFIX,
                        "tooltip": (
                            "Explicit path; when non-empty it REPLACES the "
                            "dropdown selection. Relative to ComfyUI's input "
                            "directory (output directory as fallback), or "
                            "absolute. Convert it to an input and wire the "
                            "'path' output of MOSS-TTS Save Tokens into it to "
                            "load in a later run what an earlier one wrote -- a "
                            "dropdown cannot take a link, this can."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MOSS_TOKENS", "INT")
    RETURN_NAMES = ("tokens", "frames")
    OUTPUT_TOOLTIPS = (
        "Audio codes, shape [frames, n_vq]. Must come from the same model you "
        "generate with (n_vq is validated when it is used).",
        "Number of code frames. Divide by 12.5 for seconds.",
    )
    FUNCTION = "load"
    CATEGORY = "MOSS TTS 1.5"

    @classmethod
    def VALIDATE_INPUTS(cls, path: str = "") -> bool:
        """Accept 'path' values that are not in the current dropdown.

        Naming an input here makes ComfyUI skip its built-in COMBO membership
        check for it (execution.py: `x not in validate_function_inputs`), which
        is what keeps a workflow saved before this widget became a dropdown --
        or one pointing at a file that has since moved -- loadable. The path is
        checked where it matters: when load() resolves it.
        """
        return True

    @classmethod
    def IS_CHANGED(cls, path: str, path_override: str = "") -> Any:
        """Re-run when the file itself changed, not just when the path string did."""
        try:
            stat = _resolve_tokens_path(_selected_tokens_path(path, path_override)).stat()
        except Exception:
            return float("nan")  # unreadable -> never cache, let load() raise the real error
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def load(self, path: str, path_override: str = "") -> tuple[torch.Tensor, int]:
        selected = _selected_tokens_path(path, path_override)
        resolved = _resolve_tokens_path(selected)
        try:
            payload = torch.load(str(resolved), map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - torch < 1.13 has no weights_only
            payload = torch.load(str(resolved), map_location="cpu")
        tokens = _tokens_from_payload(payload, str(resolved))
        frames = int(tokens.shape[0])
        logger.info(
            f"[MOSS-TTS] load_tokens frames={frames} seconds={_token_seconds(frames):.2f} "
            f"n_vq={int(tokens.shape[1])} "
            f"source={'path_override' if (path_override or '').strip() else 'dropdown'} "
            f"path={resolved}"
        )
        return (tokens, frames)


def _is_cjk(text: str) -> bool:
    for ch in text[:200]:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF: return True   # CJK unified ideographs
        if 0x3040 <= code <= 0x309F: return True   # hiragana
        if 0x30A0 <= code <= 0x30FF: return True   # katakana
        if 0xAC00 <= code <= 0xD7AF: return True   # hangul
    return False


class MOSSEstimateTokens:
    """Estimate MOSS's `tokens` duration hint from a text.

    Heuristic: words per minute (or characters per minute for CJK) -> seconds
    -> tokens at 12.5 frames/s. Feed the output into a MOSS Voice Clone /
    Voice Continue `target_tokens` input to steer duration.
    """

    DESCRIPTION = (
        "Rough estimator that turns text into a MOSS target-token count for "
        "duration steering. Wire the token output into a Voice Clone / Voice "
        "Continue 'target_tokens' input. NOTE: this is a heuristic (word count "
        "* wpm rate) -- if the text has heavy punctuation or long compounds, "
        "the wpm assumption drifts. Scale the output with a math node if you "
        "need to compensate."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Text to estimate. Word count via whitespace split "
                            "for space-separated languages; for CJK (Chinese, "
                            "Japanese, Korean) falls back to non-whitespace "
                            "character count."
                        ),
                    },
                ),
                "words_per_minute": (
                    "FLOAT",
                    {
                        "default": 150.0,
                        "min": 60.0,
                        "max": 400.0,
                        "step": 5.0,
                        "tooltip": (
                            "Assumed speaking rate. Reference points: 150 wpm "
                            "= calm audiobook narration, 180 wpm = "
                            "conversational, 220 wpm = fast/rushed. For CJK "
                            "this is interpreted as characters per minute. "
                            "Need slack? Chain a math node after the output."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("target_tokens",)
    OUTPUT_TOOLTIPS = (
        "Estimated MOSS 'tokens' hint. Wire into a Voice Clone / Voice "
        "Continue 'target_tokens' input. Divide by 12.5 to get seconds.",
    )
    FUNCTION = "estimate"
    CATEGORY = "MOSS TTS 1.5"

    def estimate(self, text: str, words_per_minute: float) -> tuple[int]:
        text = text.strip()
        if not text:
            return (0,)
        if _is_cjk(text):
            unit_count = sum(1 for ch in text if not ch.isspace())
        else:
            unit_count = len(text.split())
        pace_per_second = max(1e-3, float(words_per_minute) / 60.0)
        seconds = unit_count / pace_per_second
        tokens = int(math.ceil(seconds * MOSS_FRAMES_PER_SECOND))
        logger.info(
            f"[MOSS-TTS] estimate units={unit_count} wpm={words_per_minute:g} "
            f"-> seconds={seconds:.2f} tokens={tokens}"
        )
        return (tokens,)


NODE_CLASS_MAPPINGS = {
    "MOSSLoadModel": MOSSLoadModel,
    "MOSSSpeak": MOSSSpeak,
    "MOSSVoiceClone": MOSSVoiceClone,
    "MOSSVoiceContinue": MOSSVoiceContinue,
    "MOSSEstimateTokens": MOSSEstimateTokens,
    "MOSSEncodeTokens": MOSSEncodeTokens,
    "MOSSDecodeTokens": MOSSDecodeTokens,
    "MOSSConcatTokens": MOSSConcatTokens,
    "MOSSSaveTokens": MOSSSaveTokens,
    "MOSSLoadTokens": MOSSLoadTokens,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSLoadModel": "MOSS-TTS Load Model",
    "MOSSSpeak": "MOSS-TTS Speak",
    "MOSSVoiceClone": "MOSS-TTS Voice Clone",
    "MOSSVoiceContinue": "MOSS-TTS Voice Continue",
    "MOSSEstimateTokens": "MOSS-TTS Estimate Tokens",
    "MOSSEncodeTokens": "MOSS-TTS Encode Tokens",
    "MOSSDecodeTokens": "MOSS-TTS Decode Tokens",
    "MOSSConcatTokens": "MOSS-TTS Concat Tokens",
    "MOSSSaveTokens": "MOSS-TTS Save Tokens",
    "MOSSLoadTokens": "MOSS-TTS Load Tokens",
}
