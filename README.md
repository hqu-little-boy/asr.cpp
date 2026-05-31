# asr.cpp

A standalone, `whisper-cli`-style speech-recognition tool built on **llama.cpp /
mtmd**. The first target model is **Qwen3-ASR**; a per-model *profile* registry
keeps the core model-agnostic so future mtmd audio models can be added without
touching it.

## Status

Working end to end for text transcription with optional VAD segmentation.
Current outputs include stdout plus txt/json/srt/vtt/lrc/csv files. Long audio
is handled by either fixed-window chunking or FireRedVAD speech segments.

## Design

- **Public APIs only.** mtmd is a sealed dependency: the code includes
  `mtmd.h`, `mtmd-helper.h`, `llama.h`, `ggml.h`, `gguf.h` and `common/*` — never
  `clip*.h`, `mtmd-audio.h` or `models/*`.
- **Library split** — `asr_core` (pure logic: parsing, chunking, output,
  argument parsing, profile registry, merging; no llama/mtmd, fully unit-tested)
  and `asr_engine` (mtmd-backed; built when `ASR_BUILD_ENGINE=ON`). `asr-cli` is
  a thin shell.
- **Pipeline** — load audio → 16 kHz mono PCM → chunk (fixed window nudged to a
  silence dip; default 30 s) → transcribe each chunk in an *independent* context
  → merge → output.
- **Profiles** are the single extension point. A profile is selected from the
  mmproj's `clip.(audio.)projector_type` (e.g. `qwen3a`); unknown types fall back
  to a generic pass-through. Each profile defines how the prompt is built and how
  the model's raw generation is parsed. Qwen3-ASR emits
  `language <Lang><asr_text><transcription>`, which the `qwen3a` profile parses
  into `{language, text}`.

## Dependencies

- `llama.cpp/` present in the project root (vendored; built from source).
- CLI11 (system-installed) for command-line parsing.
- GoogleTest (system-installed) for unit tests.

## Build

```sh
# Core + pure-logic tests only (default; fast; does not compile llama.cpp):
cmake -S . -B build
cmake --build build -j
ctest --test-dir build

# Full build with the mtmd engine + CLI:
cmake -S . -B build-engine -DASR_BUILD_ENGINE=ON
cmake --build build-engine -j
```

## Usage

```sh
./build/asr-cli \
    -m models/Qwen3-ASR-0.6B-Q8_0.gguf \
    --mmproj models/mmproj-Qwen3-ASR-0.6B-bf16.gguf \
    audio.wav -otxt -oj
```

Common options:

| Option | Meaning |
| --- | --- |
| `-m, --model FNAME` | main GGUF model (required) |
| `--mmproj FNAME` | multimodal projector GGUF (required) |
| `-f, --file FNAME` | input audio (wav/mp3/flac); also positional |
| `-of, --output-file BASE` | output base path (writes `BASE.<format>`) |
| `-otxt` / `-oj` | write `.txt` / `.json` |
| `-osrt` / `-ovtt` | write `.srt` / `.vtt` subtitles; requires `--vad` |
| `-olrc` / `-ocsv` | write `.lrc` / `.csv`; requires `--vad` |
| `--output-format FMT` | add output format: `txt`, `json`, `srt`, `vtt`, `lrc`, or `csv` |
| `--vad, --vad-model FNAME` | segment speech with FireRedVAD |
| `--chunk-length N` | chunk length in seconds (default 30) |
| `--context TEXT` | hotword / domain bias passed to the model |
| `--profile NAME` | override the auto-detected profile |
| `-t, --threads N` | CPU threads |
| `-ng, --no-gpu` | disable GPU offload |
| `-np, --no-prints` | mute logs (transcription still prints to stdout) |
| `-n, --n-predict N` | max tokens generated per chunk |
| `-p, --processors N` | number of parallel inference instances |

The transcription always streams to `stdout`; file output flags additionally
write files. Subtitle-like formats require VAD so timestamps are available.

## Testing

`ctest` runs the pure-logic suite by default. Engine and model-dependent tests
are available in an `ASR_BUILD_ENGINE=ON` build; model-dependent cases are
skipped unless `ASR_RUN_MODEL_TESTS=1` is set and the models are present:

```sh
ASR_RUN_MODEL_TESTS=1 ctest --test-dir build-engine
```
