# asr.cpp

A standalone speech-recognition CLI built on top of **llama.cpp / mtmd**, with a
whisper-cli-like user experience. The first target model is **Qwen3-ASR**; the
design keeps a per-model *profile* registry so future mtmd audio models can be
added without touching the core.

## Status

Under active development. v1 scope: single mtmd engine, plain-text output
(`stdout` / `-otxt` / `-oj`), long-audio handling via a chunking driver. No
timestamps / subtitles yet (planned).

## Design constraints

- **Public APIs only.** mtmd is treated as a sealed dependency: we include
  `mtmd.h`, `mtmd-helper.h`, `llama.h`, `ggml.h`, `gguf.h` and `common/*` — never
  `clip*.h`, `mtmd-audio.h` or `models/*`.
- The library is split into `asr_core` (pure logic, no llama/mtmd, fully
  unit-testable) and `asr_engine` (mtmd-backed, built when `ASR_BUILD_ENGINE=ON`).

## Dependencies

- `llama.cpp/` must be present in the project root (vendored; built from source).
- GoogleTest (system-installed) for unit tests.

## Build

```sh
# Core + tests only (fast; no llama.cpp compile):
cmake -S . -B build -DASR_BUILD_ENGINE=OFF
cmake --build build
ctest --test-dir build

# Full build with the mtmd engine + CLI:
cmake -S . -B build -DASR_BUILD_ENGINE=ON
cmake --build build
```

## Usage (planned)

```sh
asr-cli -m models/Qwen3-ASR-0.6B-Q8_0.gguf \
        --mmproj models/mmproj-Qwen3-ASR-0.6B-bf16.gguf \
        audio.wav -otxt -oj
```
