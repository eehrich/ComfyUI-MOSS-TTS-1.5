"""ComfyUI nodes for OpenMOSS MOSS-TTS-Local-Transformer-v1.5.

Four nodes:
  - MOSSLoadModel:        loads the processor + model once, caches by (model_id, dtype, device).
  - MOSSVoiceClone:       MOSS_MODEL + reference AUDIO + text -> cloned AUDIO out.
  - MOSSVoiceContinue:    MOSS_MODEL + previous AUDIO + follow-up text -> continuation AUDIO out.
  - MOSSEstimateDuration: text -> (target_seconds, target_tokens) estimate for duration control.

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
DEFAULT_LANGUAGES = ("German", "English", "Chinese", "Japanese", "Korean", "French", "Spanish", "Italian")
MOSS_FRAMES_PER_SECOND = 12.5
MOSS_SAMPLE_RATE = 48000

_MODEL_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


def _load_bundle(model_id: str, device: str, dtype_name: str) -> dict[str, Any]:
    key = (model_id, device, dtype_name)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]

    logger.info(f"[MOSS-TTS] loading processor '{model_id}' ...")
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    logger.info(f"[MOSS-TTS] loading model '{model_id}' (dtype={dtype_name}, device={device}) ...")
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype,
    ).to(device)
    model.eval()

    bundle = {"processor": processor, "model": model, "device": device, "dtype": dtype}
    _MODEL_CACHE[key] = bundle
    return bundle


class MOSSLoadModel:
    """Load and cache the MOSS-TTS processor + model.

    The bundle is memoised by (model_id, device, dtype) so subsequent workflow
    runs reuse the already-loaded weights with zero overhead.
    """

    DESCRIPTION = (
        "Loads the MOSS-TTS-Local-Transformer processor + model. "
        "First execution downloads ~9 GB of weights into the Hugging Face cache "
        "and moves them to the selected device. Subsequent runs reuse the "
        "cached bundle -> no re-load penalty."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model_id": (
                    "STRING",
                    {
                        "default": DEFAULT_MODEL_ID,
                        "tooltip": (
                            "Hugging Face repo id. Defaults to the official "
                            "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5. "
                            "Any compatible fork can be plugged in here."
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
                "dtype": (
                    ["bfloat16", "float16", "float32"],
                    {
                        "default": "bfloat16",
                        "tooltip": (
                            "Weight precision. bfloat16 recommended on modern "
                            "GPUs (Ampere+); use float16 on older cards, "
                            "float32 for CPU. CPU always forces float32."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MOSS_MODEL",)
    RETURN_NAMES = ("moss_model",)
    OUTPUT_TOOLTIPS = ("Model bundle. Feed into MOSS-TTS Voice Clone or Voice Continue.",)
    FUNCTION = "load"
    CATEGORY = "MOSS TTS 1.5"

    def load(self, model_id: str, device: str, dtype: str):
        # Fall back to cpu when CUDA is missing.
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("[MOSS-TTS] CUDA requested but not available; falling back to cpu.")
            device = "cpu"
            if dtype != "float32":
                logger.warning("[MOSS-TTS] cpu path forces dtype=float32.")
                dtype = "float32"
        bundle = _load_bundle(model_id, device, dtype)
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


def _to_comfy_audio(audio_tensor: torch.Tensor) -> tuple[dict[str, Any], int, float]:
    """Return (AUDIO dict, tokens_generated, seconds)."""
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    waveform = audio_tensor.detach().cpu().unsqueeze(0)  # -> [1, C, T]
    seconds = waveform.shape[-1] / float(MOSS_SAMPLE_RATE)
    tokens_generated = int(round(seconds * MOSS_FRAMES_PER_SECOND))
    return ({"waveform": waveform, "sample_rate": MOSS_SAMPLE_RATE}, tokens_generated, seconds)


def _sanitize_target_tokens(target_tokens: int) -> int | None:
    if target_tokens and target_tokens > 0:
        return int(target_tokens)
    return None


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

            audio_tensor: torch.Tensor = processor.decode(outputs)[0].audio_codes_list[0]

        audio_dict, tokens_generated, seconds = _to_comfy_audio(audio_tensor)
        logger.info(
            f"[MOSS-TTS] clone done seconds={seconds:.2f} tokens_generated={tokens_generated}"
        )
        return (audio_dict, tokens_generated)


class MOSSVoiceContinue:
    """Continue an existing MOSS-TTS clip: same voice, more text."""

    DESCRIPTION = (
        "Extends a previously generated MOSS-TTS clip by feeding it back to "
        "the model as the 'assistant' side of a conversation and asking it "
        "to keep talking. The voice comes from the prior audio itself, so "
        "there is no separate reference audio input. Output contains only "
        "the newly-generated audio (concatenate with the input if you want "
        "the full stream)."
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
                            "Prior MOSS output to continue from. Any ComfyUI "
                            "AUDIO works (typically the AUDIO output of a "
                            "MOSS-TTS Voice Clone / Voice Continue node)."
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
                            "Text for MOSS to speak after the previous audio. "
                            "Empty is allowed: the model then improvises a "
                            "purely acoustic continuation."
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
                            "Optional target length for the CONTINUATION, in "
                            "audio frames. 0 = disabled (model decides via EOS). "
                            "12.5 frames/s -> 375 tokens ~30 s, 750 ~60 s. "
                            "Chain a MOSS-TTS Estimate Tokens node to compute "
                            "this from the follow-up text."
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
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "tokens_generated")
    OUTPUT_TOOLTIPS = (
        "The NEW audio produced after the input. Same 48 kHz stereo format.",
        "Number of audio frames the model added (frames, not samples). "
        "At 12.5 fps this equals duration_seconds * 12.5.",
    )
    FUNCTION = "generate"
    CATEGORY = "MOSS TTS 1.5"

    def generate(
        self,
        moss_model: dict[str, Any],
        previous_audio: dict[str, Any],
        text: str,
        language: str,
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
            prior_path = _comfy_audio_to_wav(previous_audio, tmp_dir, name="moss_prior.wav")

            build_kwargs: dict[str, Any] = {
                "text": text.strip(),
                "language": language,
            }
            tok_hint = _sanitize_target_tokens(target_tokens)
            if tok_hint is not None:
                build_kwargs["tokens"] = tok_hint

            user_msg = processor.build_user_message(**build_kwargs)
            assistant_msg = processor.build_assistant_message(audio_codes_list=[str(prior_path)])
            conversation = [user_msg, assistant_msg]

            logger.info(
                f"[MOSS-TTS] continue text_chars={len(build_kwargs['text'])} "
                f"lang={language} target_tokens={tok_hint or 'auto'} "
                f"prior_seconds={previous_audio['waveform'].shape[-1] / float(previous_audio['sample_rate']):.2f}"
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

            audio_tensor: torch.Tensor = processor.decode(outputs)[0].audio_codes_list[0]

        audio_dict, tokens_generated, seconds = _to_comfy_audio(audio_tensor)
        logger.info(
            f"[MOSS-TTS] continue done seconds={seconds:.2f} tokens_generated={tokens_generated}"
        )
        return (audio_dict, tokens_generated)


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
    "MOSSVoiceClone": MOSSVoiceClone,
    "MOSSVoiceContinue": MOSSVoiceContinue,
    "MOSSEstimateTokens": MOSSEstimateTokens,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSLoadModel": "MOSS-TTS Load Model",
    "MOSSVoiceClone": "MOSS-TTS Voice Clone",
    "MOSSVoiceContinue": "MOSS-TTS Voice Continue",
    "MOSSEstimateTokens": "MOSS-TTS Estimate Tokens",
}
