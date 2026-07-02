# ComfyUI-MOSS-TTS-1.5

ComfyUI custom nodes for [**MOSS-TTS-Local-Transformer-v1.5**](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) by [OpenMOSS](https://github.com/OpenMOSS).
Four lean nodes for zero-shot voice cloning, deterministic duration steering, and audio continuation — no fine-tuning, no separate reference-transcript dance, just an audio reference and a target text.

- **31 languages** (with explicit language tag support)
- **48 kHz stereo** output
- **Zero-shot voice cloning** from a single reference clip
- **Hard duration control** via `target_tokens` (empirically verified — MOSS obeys it precisely)
- **Continuation mode** — extend a previously generated clip in the same voice
- **Text → token estimator** so the token count doesn't have to be a guess

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

All four nodes live under the top-level **`MOSS TTS 1.5`** category in the ComfyUI menu.

### `MOSS-TTS Load Model`

Loads the processor + model and caches the instance in-memory across runs.
Subsequent workflow queues re-use the already-loaded model — no re-load penalty.

| Input | Type | Default | Notes |
|---|---|---|---|
| `model_id` | STRING | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` | Any compatible Hugging Face repo id |
| `device` | `cuda` \| `cpu` | `cuda` | Falls back to `cpu` when CUDA is unavailable |
| `dtype` | `bfloat16` \| `float16` \| `float32` | `bfloat16` | `cpu` forces `float32` |

**Output**: `MOSS_MODEL` — pass to Voice Clone or Voice Continue.

### `MOSS-TTS Voice Clone`

Generates speech from `text` in the voice of `reference_audio`.

| Input | Type | Default | Notes |
|---|---|---|---|
| `moss_model` | MOSS_MODEL | — | From the loader |
| `reference_audio` | AUDIO | — | ComfyUI `AUDIO` type (`LoadAudio`, another node's output, etc.) |
| `text` | STRING | `Hello, this is a test.` | Multiline |
| `language` | enum | `English` | 31 supported: German, English, Chinese, Japanese, Korean, French, Spanish, Italian, and more (see [MOSS README](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)) |
| `instruction` | STRING | `""` | Optional free-form style/direction hint. **Not** a reference transcript — MOSS has no reference-text channel. |
| `audio_temperature` | FLOAT | `1.7` | Sampling temperature |
| `audio_top_p` | FLOAT | `0.8` | Nucleus sampling |
| `audio_top_k` | INT | `25` | Top-k sampling |
| `target_tokens` | INT | `0` | Target duration in audio frames (12.5 fps → 375 ≈ 30 s, 750 ≈ 60 s). `0` = disabled, model decides via EOS. See [Duration control](#duration-control). |
| `max_new_tokens` | INT | `4096` | Safety cap on generated audio frames. MOSS treats this as its internal `frame_budget` at 12.5 fps → default `4096` caps output at ~5 min. |
| `seed` | INT | `42` | Random seed. Same seed + same inputs → identical output. |

**Outputs**:

- `audio` — 48 kHz stereo AUDIO, ready for `PreviewAudio` / `SaveAudio`
- `tokens_generated` — INT, number of audio frames actually produced (divide by 12.5 for seconds)

### `MOSS-TTS Voice Continue`

Extends a previously generated MOSS clip: the model treats the input audio as the assistant's prior turn (`mode="continuation"`) and keeps talking. The voice comes from the input clip itself, so there's no separate reference audio.

| Input | Type | Default | Notes |
|---|---|---|---|
| `moss_model` | MOSS_MODEL | — | From the loader |
| `previous_audio` | AUDIO | — | Prior MOSS output (typically another Voice Clone / Voice Continue node's `audio` output) |
| `text` | STRING | `""` | Follow-up text. Empty is allowed → purely acoustic continuation. |
| `language` | enum | `English` | Same list as Voice Clone |
| `audio_temperature` | FLOAT | `1.7` | Sampling temperature |
| `audio_top_p` | FLOAT | `0.8` | Nucleus sampling |
| `audio_top_k` | INT | `25` | Top-k sampling |
| `target_tokens` | INT | `0` | Duration of the **new** segment in frames. `0` = model decides via EOS. |
| `max_new_tokens` | INT | `4096` | Safety cap on the new segment |
| `seed` | INT | `42` | Random seed |

**Outputs**: same shape as Voice Clone — `audio` (only the newly-generated segment, not concatenated with input) and `tokens_generated`.

### `MOSS-TTS Estimate Tokens`

Turns a text into a `target_tokens` estimate you can wire straight into `Voice Clone` / `Voice Continue`.

| Input | Type | Default | Notes |
|---|---|---|---|
| `text` | STRING | `""` | Multiline. Word count via whitespace split; CJK (Chinese/Japanese/Korean) falls back to non-whitespace character count. |
| `words_per_minute` | FLOAT | `150.0` | 150 = calm audiobook narration, 180 = conversational, 220 = fast. For CJK read as characters-per-minute. |

**Output**: `target_tokens` (INT). Formula: `ceil(word_count / (wpm/60) * 12.5)`.

Need slack for punctuation-heavy passages? Chain a ComfyUI math node (`Multiply` / `Add`) after the output — the estimator deliberately has no built-in buffer so you can compose one that scales with the text.

---

## Duration control

MOSS's `build_user_message` accepts a `tokens` field (in audio frames, 12.5 fps). Empirically **MOSS obeys this precisely** — same text with `target_tokens = 100, 200, 400` produces audio of roughly `8, 16, 32 s`. This nodepack exposes it as `target_tokens` on both `Voice Clone` and `Voice Continue`.

Practical uses:

- **Consistent narration pace across a batch**: fix `wpm = 150` in `Estimate Tokens`, MOSS will read every chapter at the same tempo regardless of length.
- **Speech-rate control without style prompting**: chain a multiplier after the estimator. `× 1.4` = slow / dramatic, `× 0.75` = urgent / rushed. Cleaner than adjectives in the `instruction` field.
- **Fixed video/audio slots**: your video shot is 8 s → set `target_tokens = 100`. MOSS fits into that slot.
- **Continuation length steering**: `Voice Continue.target_tokens = 375` → about 30 s of extra audio.

`max_new_tokens` is a separate parameter — a hard cap on `frame_budget` in MOSS's generation loop (see `modeling_moss_tts.py`: `frame_budget = max_new_frames if max_new_frames is not None else max_new_tokens`). Keep it comfortably above `target_tokens` as a runaway fuse; the default `4096` (~5 min at 12.5 fps) is usually plenty.

---

## Example workflows

**Basic voice clone with automatic duration:**

```
[Load Audio]        [MOSS-TTS Load Model]
      \                  /
       > [MOSS-TTS Voice Clone] -> [Save Audio]
                 ^
    [text]  [MOSS-TTS Estimate Tokens] -> target_tokens
```

**Speech-rate control:**

```
[text] -> [Estimate Tokens] -> [Multiply INT × 1.4] -> Voice Clone.target_tokens
```

Same audio reference, same seed, same text — but 40% slower / more dramatic. Or `× 0.75` for urgent.

**Continuation chain:**

```
[LoadAudio] -> [Voice Clone] -> audio ─┐
[part 1 text] -> Voice Clone.text      └─> [Voice Continue] -> [Save Audio]
                                            ^        ^
                                    [part 2 text]  [Estimate Tokens for part 2]
```

**Or as a standalone Python demo** (what the nodes wrap under the hood):

```python
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
    tokens=125,   # ~10 s target
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

## Performance & memory

Measured on a single-turn 75-character German sentence, RTX 5090 (bf16):

| Phase | Time |
|---|---|
| Processor load (audio tokenizer moved to GPU) | ~21 s |
| Model load (9.1 GB checkpoint → GPU) | ~16 s |
| Generation (4.72 s of audio) | **~2.7 s** |

Load happens once per (model_id, device, dtype). Warm-cache generation is real-time on modern hardware.

VRAM: ~10 GB active weight + activations in `bfloat16`. Peak spikes with long contexts (e.g. very long text or `max_new_tokens=16384`) can push higher. RTX 3090 (24 GB) has comfortable headroom.

---

## Troubleshooting

- **`Can't load the model … pytorch_model.bin`**: your model.safetensors download stalled. Re-run `huggingface_hub.hf_hub_download(repo_id=..., filename="model.safetensors")` explicitly. Often caused by low disk space in `~/.cache/huggingface`.
- **`std::bad_alloc` on `import torchcodec`**: your installed `torchcodec` version was compiled against a different torch. Either match versions (torchcodec 0.8.x with torch 2.8.x, 0.9.x with 2.9.x, 0.10.x with 2.10.x) or `pip uninstall torchcodec`. The MOSS pipeline itself does **not** require torchcodec.
- **`build_user_message() got an unexpected keyword argument 'reference_text'`**: fixed in `0.1.1` — MOSS has no reference-text channel. Use `instruction` for style hints, or rely on `reference` (audio) + `language` alone.
- **Text like `[pause 1.2s]` is spoken as literal words**: MOSS v1.5 has no built-in pause-marker parser (verified against the source — no `pause`/`silence` tokens in `added_tokens.json`, no bracketed-marker regex in `processing_moss_tts.py`). For deterministic gaps, generate two clips and concatenate with a silence spacer in ComfyUI, or use `Voice Continue` in a chain.

---

## License

- **This nodepack**: [MIT](./LICENSE)
- **MOSS-TTS-Local-Transformer-v1.5 model & code**: [Apache 2.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5), copyright OpenMOSS-Team.

## Credits

- Model: [OpenMOSS-Team](https://github.com/OpenMOSS) / [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- Wrapper: this repo — a thin bridge to ComfyUI's `AUDIO` type and its category tree.

Not affiliated with OpenMOSS. Star the [upstream model](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) if you like the work.
