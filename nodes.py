"""ComfyUI nodes for OpenMOSS MOSS-TTS-Local-Transformer-v1.5.

Two nodes:
  - MOSSLoadModel:  loads the processor + model once, caches by (model_id, dtype, device).
  - MOSSVoiceClone: uses a MOSS_MODEL bundle + a reference AUDIO + text -> cloned AUDIO out.

ComfyUI AUDIO shape: {"waveform": Tensor[B, C, T], "sample_rate": int}. We convert to a
temp WAV file for the MOSS processor (which takes file paths), then convert back on output.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import torch
import torchaudio

logger = logging.getLogger("MOSS-TTS-ComfyUI")

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
DEFAULT_LANGUAGES = ("German", "English", "Chinese", "Japanese", "Korean", "French", "Spanish", "Italian")

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
    OUTPUT_TOOLTIPS = ("Model bundle. Feed into MOSS-TTS Voice Clone.",)
    FUNCTION = "load"
    CATEGORY = "audio/MOSS-TTS"

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


def _comfy_audio_to_wav(audio: dict[str, Any], tmp_dir: Path) -> Path:
    """Persist a ComfyUI AUDIO dict to a temporary WAV file the MOSS processor can read."""
    waveform: torch.Tensor = audio["waveform"]
    sample_rate: int = int(audio["sample_rate"])
    if waveform.dim() == 3:
        # ComfyUI AUDIO is [B, C, T] -- take the first batch
        waveform = waveform[0]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [T] -> [1, T]
    out = tmp_dir / "moss_ref.wav"
    torchaudio.save(str(out), waveform.detach().cpu(), sample_rate)
    return out


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
                            "Text to synthesize in the cloned voice. Supports "
                            "inline pause markers like [pause 1.2s] for "
                            "deterministic silences."
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
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 65536,
                        "step": 128,
                        "tooltip": (
                            "Upper bound on generated audio tokens. MOSS has "
                            "12 codebooks and runs at 12.5 frames/sec, so "
                            "150 tokens = 1 s of audio. Default 4096 = ~27 s "
                            "(fine for a sentence or two). Raise for longer "
                            "text: 9000 = ~1 min, 45000 = ~5 min."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "tooltip": (
                            "Random seed. Same seed + same inputs = identical "
                            "audio output (deterministic replay)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_TOOLTIPS = ("Generated audio at 48 kHz stereo, ready for SaveAudio / PreviewAudio.",)
    FUNCTION = "generate"
    CATEGORY = "audio/MOSS-TTS"

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
        max_new_tokens: int,
        seed: int,
    ) -> tuple[dict[str, Any]]:
        processor = moss_model["processor"]
        model = moss_model["model"]
        device = moss_model["device"]

        # Deterministic sampling via seeded generator.
        torch.manual_seed(int(seed))
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(int(seed))

        with tempfile.TemporaryDirectory(prefix="moss_") as td:
            tmp_dir = Path(td)
            ref_path = _comfy_audio_to_wav(reference_audio, tmp_dir)

            build_kwargs: dict[str, Any] = {
                "text": text.strip(),
                "reference": [str(ref_path)],
                "language": language,
            }
            if instruction.strip():
                build_kwargs["instruction"] = instruction.strip()

            logger.info(
                f"[MOSS-TTS] generating text_chars={len(build_kwargs['text'])} "
                f"lang={language} instruction={'set' if instruction.strip() else 'none'} "
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

        # audio_tensor is [C, T] at 48 kHz per model spec -- wrap for ComfyUI AUDIO.
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        waveform = audio_tensor.detach().cpu().unsqueeze(0)  # -> [1, C, T]

        duration = waveform.shape[-1] / 48000.0
        logger.info(f"[MOSS-TTS] generated {duration:.2f}s @48kHz shape={tuple(waveform.shape)}")
        return ({"waveform": waveform, "sample_rate": 48000},)


NODE_CLASS_MAPPINGS = {
    "MOSSLoadModel": MOSSLoadModel,
    "MOSSVoiceClone": MOSSVoiceClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSLoadModel": "MOSS-TTS Load Model",
    "MOSSVoiceClone": "MOSS-TTS Voice Clone",
}
