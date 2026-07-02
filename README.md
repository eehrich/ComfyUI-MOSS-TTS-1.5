# ComfyUI-MOSS-TTS-1.5

ComfyUI custom nodes for [**MOSS-TTS-Local-Transformer-v1.5**](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) by [OpenMOSS](https://github.com/OpenMOSS).
Two lean nodes for zero-shot voice cloning and multilingual TTS — no fine-tuning, no separate reference-transcript dance, just an audio reference and a target text.

- **31 languages** (with explicit language tag support)
- **48 kHz stereo** output
- **Zero-shot voice cloning** from a single reference clip
- **Inline pause markers** in text: `"[pause 1.2s]"`
- **Explicit duration control** via token budget

The model itself is Apache-2.0 released by OpenMOSS-Team. This nodepack is MIT.

---

## Requirements

- ComfyUI running on a machine with a CUDA GPU (~10 GB VRAM in `bfloat16`)
- Python 3.10+
- `transformers >= 5.0.0` (v5.5.x recommended; the same range MOSS's official code targets)
- `torch`, `torchaudio` (whatever your ComfyUI already ships with)
- ~9.1 GB free disk for the model weights (auto-downloaded from Hugging Face)

That's it — no extra CUDA extensions, no custom kernels.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/eehrich/ComfyUI-MOSS-TTS-1.5.git MOSS-TTS-ComfyUI
```

Restart ComfyUI. The first `MOSS-TTS Load Model` execution will download the checkpoint (~9.1 GB) into your Hugging Face cache.

### Known install gotcha — `configuration_moss_audio_tokenizer.py` dataclass ordering

On Python 3.11+ the auto-downloaded `configuration_moss_audio_tokenizer.py` from
`OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` declares its dataclass fields without defaults **after** the
parent class already added defaulted fields, so `dataclass(...)` raises:

```
TypeError: non-default argument 'sampling_rate' follows default argument 'problem_type'
```

Fix once, after the first failed load, in the auto-downloaded file under
`~/.cache/huggingface/modules/transformers_modules/OpenMOSS_hyphen_Team/MOSS_hyphen_Audio_hyphen_Tokenizer_hyphen_v2/<hash>/configuration_moss_audio_tokenizer.py`
— give each of these class fields a `= None` default:

```python
sampling_rate: int = None
downsample_rate: int = None
causal_transformer_context_duration: float = None
encoder_kwargs: list[dict[str, Any]] = None
decoder_kwargs: list[dict[str, Any]] = None
number_channels: int = None
enable_channel_interleave: bool = None
attention_implementation: str = None
compute_dtype: str = None
codec_weight_dtype: str = None
quantizer_type: str = None
quantizer_kwargs: dict[str, Any] = None
```

Nothing behavioural changes — the real defaults still come from the class's `__init__`.

---

## Nodes

Both nodes live under the **`audio/MOSS-TTS`** category.

### `MOSS-TTS Load Model`

Loads the processor + model and caches the instance in-memory across runs.
Subsequent workflow queues re-use the already-loaded model — no re-load penalty.

| Input | Type | Default | Notes |
|---|---|---|---|
| `model_id` | STRING | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` | Any compatible Hugging Face repo id |
| `device` | `cuda` \| `cpu` | `cuda` | Falls back to `cpu` when CUDA is unavailable |
| `dtype` | `bfloat16` \| `float16` \| `float32` | `bfloat16` | `cpu` forces `float32` |

**Output**: `MOSS_MODEL` — pass to Voice Clone.

### `MOSS-TTS Voice Clone`

Generates speech from `text` in the voice of `reference_audio`.

| Input | Type | Default | Notes |
|---|---|---|---|
| `moss_model` | MOSS_MODEL | — | From the loader |
| `reference_audio` | AUDIO | — | ComfyUI `AUDIO` type (`LoadAudio`, another node's output, etc.) |
| `text` | STRING | `Hallo, das ist ein Test.` | Multiline. Supports inline `[pause 1.2s]` markers |
| `language` | enum | `German` | 31 supported: German, English, Chinese, Japanese, Korean, French, Spanish, Italian, and more (see [MOSS README](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)) |
| `instruction` | STRING | `""` | Optional free-form style/direction hint. **Not** a reference transcript — MOSS has no reference-text channel. |
| `audio_temperature` | FLOAT | `1.7` | Sampling temperature |
| `audio_top_p` | FLOAT | `0.8` | Nucleus sampling |
| `audio_top_k` | INT | `25` | Top-k sampling |
| `max_new_tokens` | INT | `4096` | ~12.5 tokens/sec so 4096 ≈ 5 min headroom |
| `seed` | INT | `42` | Set both `torch` and CUDA seeds for reproducible takes |

**Output**: `AUDIO` (48 kHz stereo). Feed into `PreviewAudio` or `SaveAudio`.

---

## Example workflow

```
┌────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌────────────┐
│ LoadAudio  ├──▶ │ MOSS-TTS Load    ├──▶ │ MOSS-TTS Voice   ├──▶ │ SaveAudio  │
│ voice.wav  │    │ Model            │    │ Clone            │    └────────────┘
└────────────┘    │  cuda / bf16     │    │  text = "..."    │
                  └──────────────────┘    │  language = DE   │
                                          └──────────────────┘
```

Or as an inline demo:

```python
# equivalent standalone (what the nodes wrap):
from transformers import AutoModel, AutoProcessor
import torch, torchaudio

processor = AutoProcessor.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    trust_remote_code=True,
)
processor.audio_tokenizer = processor.audio_tokenizer.to("cuda")

model = AutoModel.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda")

conv = [processor.build_user_message(
    text="Der Wind hörte auf, noch bevor Tessa Brandt den Grund der Senke erreichte.",
    reference=["voice.wav"],
    language="German",
)]
batch = processor([conv], mode="generation")
out = model.generate(
    input_ids=batch["input_ids"].to("cuda"),
    attention_mask=batch["attention_mask"].to("cuda"),
    max_new_tokens=4096,
    audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25,
)
audio = processor.decode(out)[0].audio_codes_list[0]
torchaudio.save("out.wav", audio.cpu(), 48000)
```

---

## Duration control

MOSS emits at ~12.5 tokens/sec. If you want a fixed length, pass `tokens=…` to the processor's `build_user_message` — 125 tokens ≈ 10 seconds. This nodepack doesn't expose that as a required knob yet, but you can fork and add it in ~5 lines. PRs welcome.

## Inline pauses

Put a marker directly in the transcript:

```
Der Wind hörte auf, [pause 1.4s] noch bevor sie den Grund der Senke erreichte.
```

MOSS honours the pause deterministically instead of relying on sampled prosody.

---

## Performance & memory

Measured on a single-turn 75-character German sentence, RTX 5090 (bf16):

| Phase | Time |
|---|---|
| Processor load (audio tokenizer moved to GPU) | ~21 s |
| Model load (9.1 GB checkpoint → GPU) | ~16 s |
| Generation (4.72 s of audio) | **~2.7 s** |

Load happens once per (model_id, device, dtype). Warm-cache generation is real-time on modern hardware.

VRAM: ~10 GB active weight + activations in `bfloat16`. Peak spikes with long contexts (e.g. very long text or `max_new_tokens=16384`) can push higher.

---

## Troubleshooting

- **`Can't load the model … pytorch_model.bin`**: your model.safetensors download stalled. Re-run `huggingface_hub.hf_hub_download(repo_id=..., filename="model.safetensors")` explicitly. Often caused by low disk space in `~/.cache/huggingface`.
- **`std::bad_alloc` on `import torchcodec`**: your installed `torchcodec` version was compiled against a different torch. Either match versions (torchcodec 0.8.x with torch 2.8.x, 0.9.x with 2.9.x, 0.10.x with 2.10.x) or `pip uninstall torchcodec`. The MOSS pipeline itself does **not** require torchcodec.
- **`build_user_message() got an unexpected keyword argument 'reference_text'`**: fixed in `0.1.1` — MOSS has no reference-text channel. Use `instruction` for style hints, or rely on `reference` (audio) + `language` alone.

---

## License

- **This nodepack**: [MIT](./LICENSE)
- **MOSS-TTS-Local-Transformer-v1.5 model & code**: [Apache 2.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5), copyright OpenMOSS-Team.

## Credits

- Model: [OpenMOSS-Team](https://github.com/OpenMOSS) / [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- Wrapper: this repo — a thin bridge to ComfyUI's `AUDIO` type and its category tree.

Not affiliated with OpenMOSS. Star the [upstream model](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) if you like the work.
