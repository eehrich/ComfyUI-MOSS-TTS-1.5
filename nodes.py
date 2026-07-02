"""ComfyUI nodes for OpenMOSS MOSS-TTS-Local-Transformer-v1.5.

Five nodes:
  - MOSSLoadModel:        loads the processor + model once, caches by (model_id, device).
  - MOSSSpeak:            MOSS_MODEL + text (no reference) -> AUDIO in a MOSS default voice.
  - MOSSVoiceClone:       MOSS_MODEL + reference AUDIO + text -> cloned AUDIO out.
  - MOSSVoiceContinue:    MOSS_MODEL + previous AUDIO + follow-up text -> continuation AUDIO out.
  - MOSSEstimateTokens:   text -> target_tokens estimate for duration control.

ComfyUI AUDIO shape: {"waveform": Tensor[B, C, T], "sample_rate": int}. We convert to a
temp WAV file for the MOSS processor (which takes file paths), then convert back on output.
"""

from __future__ import annotations

import logging
import math
import tempfile
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

_MODEL_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _resolve_dtype(device: str) -> tuple[torch.dtype, str]:
    """bf16 on CUDA (MOSS's training precision), fp32 on CPU (bf16 CPU kernels are patchy)."""
    if device.startswith("cuda"):
        return torch.bfloat16, "bfloat16"
    return torch.float32, "float32"


def _load_bundle(model_id: str, device: str) -> dict[str, Any]:
    key = (model_id, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    dtype, dtype_name = _resolve_dtype(device)

    logger.info(f"[MOSS-TTS] loading processor '{model_id}' ...")
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    logger.info(f"[MOSS-TTS] loading model '{model_id}' (dtype={dtype_name}, device={device}) ...")
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype,
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
        }

    RETURN_TYPES = ("MOSS_MODEL",)
    RETURN_NAMES = ("moss_model",)
    OUTPUT_TOOLTIPS = ("Model bundle. Feed into any MOSS-TTS Speak / Voice Clone / Voice Continue node.",)
    FUNCTION = "load"
    CATEGORY = "MOSS TTS 1.5"

    def load(self, model_id: str, device: str):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("[MOSS-TTS] CUDA requested but not available; falling back to cpu.")
            device = "cpu"
        repo_id = _repo_id_from_label(model_id)
        bundle = _load_bundle(repo_id, device)
        return (bundle,)


def _comfy_audio_to_wav(audio: dict[str, Any], tmp_dir: Path, name: str = "moss_ref.wav") -> Path:
    """Persist a ComfyUI AUDIO dict to a temporary WAV file the MOSS processor can read."""
    waveform: torch.Tensor = audio["waveform"]
    sample_rate: int = int(audio["sample_rate"])
    if waveform.dim() == 3:
        # ComfyUI AUDIO is [B, C, T] -- take the first batch
        waveform = waveform[0]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [T] -> [1, T]
    out = tmp_dir / name
    torchaudio.save(str(out), waveform.detach().cpu(), sample_rate)
    return out


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


class MOSSSpeak:
    """Text-to-speech without a voice reference (MOSS's built-in 'None' voice path)."""

    DESCRIPTION = (
        "Generates speech without a reference audio. MOSS was trained on a "
        "'None' placeholder path (see processing_moss_tts.py: else-branch of "
        "_build_generation_or_voice_clone_codes) that lets it pick a voice "
        "based on language + the 'instruction' hint. Since there is no audio "
        "reference here, 'instruction' is the ONLY voice-steering knob -- "
        "worth trying things like 'male, warm, elderly narrator' or 'young "
        "female, cheerful, energetic'."
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
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated")
    OUTPUT_TOOLTIPS = (
        "Generated audio at 48 kHz stereo, ready for SaveAudio / PreviewAudio.",
        "Number of audio frames MOSS actually generated (frames, not samples). "
        "At 12.5 fps this equals duration_seconds * 12.5.",
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
    ) -> tuple[dict[str, Any], int]:
        processor = moss_model["processor"]
        model = moss_model["model"]
        device = moss_model["device"]
        _seed(device, seed)

        build_kwargs: dict[str, Any] = {
            "text": text.strip(),
            "language": language,
        }
        if instruction.strip():
            build_kwargs["instruction"] = instruction.strip()
        tok_hint = _sanitize_target_tokens(target_tokens)
        if tok_hint is not None:
            build_kwargs["tokens"] = tok_hint

        logger.info(
            f"[MOSS-TTS] speak text_chars={len(build_kwargs['text'])} "
            f"lang={language} instruction={'set' if instruction.strip() else 'none'} "
            f"target_tokens={tok_hint or 'auto'} "
            f"temperature={audio_temperature} top_p={audio_top_p} top_k={audio_top_k}"
        )
        conversation = [processor.build_user_message(**build_kwargs)]
        batch = processor([conversation], mode="generation")

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=int(max_new_tokens),
                audio_temperature=float(audio_temperature),
                audio_top_p=float(audio_top_p),
                audio_top_k=int(audio_top_k),
            )

        audio_tensor: torch.Tensor = _extract_audio(processor, outputs)
        sample_rate = int(moss_model["sample_rate"])
        audio_dict, tokens_generated, seconds = _to_comfy_audio(audio_tensor, sample_rate)
        logger.info(
            f"[MOSS-TTS] speak done seconds={seconds:.2f} "
            f"tokens_generated={tokens_generated} sample_rate={sample_rate}"
        )
        return (audio_dict, tokens_generated)


class MOSSVoiceClone:
    """Generate speech in the cloned voice from a reference AUDIO."""

    DESCRIPTION = (
        "Zero-shot voice cloning. Feed a reference audio clip (any length, "
        "any language supported by MOSS) plus target text and MOSS returns "
        "the target text spoken in that voice at 48 kHz stereo. "
        "MOSS does NOT accept a reference transcript -- only the audio "
        "clip and an optional style hint via the 'instruction' input."
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
                "reference_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Voice reference. Any ComfyUI AUDIO source works "
                            "(LoadAudio, another node's output, etc.). Short "
                            "5-15 s clips are usually best; MOSS v1.5 also "
                            "handles long references reliably."
                        ),
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
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated")
    OUTPUT_TOOLTIPS = (
        "Generated audio at 48 kHz stereo, ready for SaveAudio / PreviewAudio.",
        "Number of audio frames MOSS actually generated (frames, not samples). "
        "At 12.5 fps this equals duration_seconds * 12.5.",
    )
    FUNCTION = "generate"
    CATEGORY = "MOSS TTS 1.5"

    def generate(
        self,
        moss_model: dict[str, Any],
        reference_audio: dict[str, Any],
        text: str,
        language: str,
        instruction: str,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        target_tokens: int,
        max_new_tokens: int,
        seed: int,
    ) -> tuple[dict[str, Any], int]:
        processor = moss_model["processor"]
        model = moss_model["model"]
        device = moss_model["device"]
        _seed(device, seed)

        with tempfile.TemporaryDirectory(prefix="moss_") as td:
            tmp_dir = Path(td)
            ref_path = _comfy_audio_to_wav(reference_audio, tmp_dir, name="moss_ref.wav")

            build_kwargs: dict[str, Any] = {
                "text": text.strip(),
                "reference": [str(ref_path)],
                "language": language,
            }
            if instruction.strip():
                build_kwargs["instruction"] = instruction.strip()
            tok_hint = _sanitize_target_tokens(target_tokens)
            if tok_hint is not None:
                build_kwargs["tokens"] = tok_hint

            logger.info(
                f"[MOSS-TTS] clone text_chars={len(build_kwargs['text'])} "
                f"lang={language} instruction={'set' if instruction.strip() else 'none'} "
                f"target_tokens={tok_hint or 'auto'} "
                f"temperature={audio_temperature} top_p={audio_top_p} top_k={audio_top_k}"
            )
            conversation = [processor.build_user_message(**build_kwargs)]
            batch = processor([conversation], mode="generation")

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    max_new_tokens=int(max_new_tokens),
                    audio_temperature=float(audio_temperature),
                    audio_top_p=float(audio_top_p),
                    audio_top_k=int(audio_top_k),
                )

            audio_tensor: torch.Tensor = _extract_audio(processor, outputs)

        sample_rate = int(moss_model["sample_rate"])
        audio_dict, tokens_generated, seconds = _to_comfy_audio(audio_tensor, sample_rate)
        logger.info(
            f"[MOSS-TTS] clone done seconds={seconds:.2f} "
            f"tokens_generated={tokens_generated} sample_rate={sample_rate}"
        )
        return (audio_dict, tokens_generated)


class MOSSVoiceContinue:
    """Continue an existing MOSS-TTS clip: same voice, more text."""

    DESCRIPTION = (
        "Extends a previously generated MOSS-TTS clip. Internally MOSS is a "
        "PREFIX-continuation model -- it needs the ORIGINAL text that "
        "produced 'previous_audio' so it can lock onto the exact point in "
        "the script where the audio left off, then produce audio for the "
        "follow-up 'text'. This node concatenates 'previous_text' + 'text' "
        "into the full script MOSS conditions on; the voice comes from the "
        "prior audio itself (no separate reference input needed). Output "
        "contains only the newly-generated audio (concatenate with the "
        "input if you want the full stream)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "moss_model": (
                    "MOSS_MODEL",
                    {"tooltip": "Model bundle produced by MOSS-TTS Load Model."},
                ),
                "previous_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Prior MOSS output to continue from. Typically the "
                            "AUDIO output of a preceding MOSS-TTS Voice Clone / "
                            "Voice Continue node. Must be paired with the exact "
                            "'previous_text' that produced it."
                        ),
                    },
                ),
                "previous_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "The exact text that produced 'previous_audio'",
                        "tooltip": (
                            "The exact text that produced 'previous_audio'. "
                            "MOSS needs it to align its 'where am I in the "
                            "script?' state. Should match word-for-word "
                            "(punctuation matters). Passing the wrong prior "
                            "text -> Kauderwelsch."
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
                            "rounding)."
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
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated", "full_audio", "full_tokens")
    OUTPUT_TOOLTIPS = (
        "New segment only, head-trimmed. Use this for per-segment QC / "
        "preview -- you hear just the delta MOSS produced this call.",
        "Frames of the NEW segment only (frames, not samples). At 12.5 fps "
        "this equals duration_seconds * 12.5.",
        "Cumulative audio: previous_audio + new segment concatenated at "
        "48 kHz stereo. Wire this into the NEXT Continue's previous_audio "
        "when the same speaker keeps talking across segments.",
        "Cumulative frame count: prefix + new. Wire into the next Continue's "
        "previous_tokens for a precise handoff without re-measurement.",
    )
    FUNCTION = "generate"
    CATEGORY = "MOSS TTS 1.5"

    def generate(
        self,
        moss_model: dict[str, Any],
        previous_audio: dict[str, Any],
        previous_text: str,
        text: str,
        language: str,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        target_tokens: int,
        max_new_tokens: int,
        seed: int,
        previous_tokens: int = 0,
        head_trim_frames: int = 1,
    ) -> tuple[dict[str, Any], int, dict[str, Any], int]:
        processor = moss_model["processor"]
        model = moss_model["model"]
        device = moss_model["device"]
        _seed(device, seed)

        prev = previous_text.strip()
        new = text.strip()
        full_text = (prev + " " + new).strip() if prev else new

        prior_seconds = previous_audio["waveform"].shape[-1] / float(previous_audio["sample_rate"])
        if previous_tokens and previous_tokens > 0:
            prefix_frames = int(previous_tokens)
            prefix_source = "wired"
        else:
            prefix_frames = int(round(prior_seconds * MOSS_FRAMES_PER_SECOND))
            prefix_source = "measured"

        with tempfile.TemporaryDirectory(prefix="moss_") as td:
            tmp_dir = Path(td)
            prior_path = _comfy_audio_to_wav(previous_audio, tmp_dir, name="moss_prior.wav")

            build_kwargs: dict[str, Any] = {
                "text": full_text,
                "language": language,
            }
            tok_hint = _sanitize_target_tokens(target_tokens)
            total_tokens = None
            if tok_hint is not None:
                # MOSS interprets `tokens` in continuation as TOTAL (prefix + new).
                # The node's `target_tokens` input is defined as "frames of NEW audio",
                # so we add the measured prefix here.
                total_tokens = tok_hint + prefix_frames
                build_kwargs["tokens"] = total_tokens

            user_msg = processor.build_user_message(**build_kwargs)
            assistant_msg = processor.build_assistant_message(audio_codes_list=[str(prior_path)])
            conversation = [user_msg, assistant_msg]

            logger.info(
                f"[MOSS-TTS] continue prev_chars={len(prev)} new_chars={len(new)} "
                f"full_chars={len(full_text)} lang={language} "
                f"target_new={tok_hint or 'auto'} "
                f"prefix_frames={prefix_frames} ({prefix_source}) "
                f"total_tokens_to_moss={total_tokens or 'auto'} prior_seconds={prior_seconds:.2f}"
            )
            batch = processor([conversation], mode="continuation")

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    max_new_tokens=int(max_new_tokens),
                    audio_temperature=float(audio_temperature),
                    audio_top_p=float(audio_top_p),
                    audio_top_k=int(audio_top_k),
                )

            audio_tensor: torch.Tensor = _extract_audio(processor, outputs)

        sample_rate = int(moss_model["sample_rate"])
        trim = max(0, int(head_trim_frames))
        if trim > 0 and audio_tensor.numel() > 0:
            # samples_per_frame = sample_rate / frames_per_second (48000/12.5=3840 for 1.7B, 24000/12.5=1920 for 8B)
            trim_samples = int(round(trim * sample_rate / MOSS_FRAMES_PER_SECOND))
            trim_samples = min(trim_samples, audio_tensor.shape[-1] - 1)
            if trim_samples > 0:
                audio_tensor = audio_tensor[..., trim_samples:]

        audio_dict, tokens_generated, seconds = _to_comfy_audio(audio_tensor, sample_rate)

        full_audio_dict, full_tokens = _concat_full_audio(
            previous_audio, audio_dict, prefix_frames, tokens_generated, sample_rate
        )

        logger.info(
            f"[MOSS-TTS] continue done seconds={seconds:.2f} "
            f"tokens_generated={tokens_generated} head_trim_frames={trim} "
            f"full_seconds={full_audio_dict['waveform'].shape[-1]/sample_rate:.2f} "
            f"full_tokens={full_tokens} sample_rate={sample_rate}"
        )
        return (audio_dict, tokens_generated, full_audio_dict, full_tokens)


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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSLoadModel": "MOSS-TTS Load Model",
    "MOSSSpeak": "MOSS-TTS Speak",
    "MOSSVoiceClone": "MOSS-TTS Voice Clone",
    "MOSSVoiceContinue": "MOSS-TTS Voice Continue",
    "MOSSEstimateTokens": "MOSS-TTS Estimate Tokens",
}
