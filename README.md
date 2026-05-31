# asr.cpp

> **English** | [中文](#中文)

A standalone, `whisper-cli`-style speech-recognition tool built on **llama.cpp / mtmd**. The first target model is **Qwen3-ASR**; a per-model *profile* registry keeps the core model-agnostic so future mtmd audio models can be added without touching it.

## Status

Working end to end for text transcription with optional VAD segmentation. Current outputs include stdout plus txt/json/srt/vtt/lrc/csv files. Long audio is handled by either fixed-window chunking or FireRedVAD speech segments.

## Supported Models

| Model | Size | Download |
| --- | --- | --- |
| Qwen3-ASR-1.7B | ~1.7 GB (Q8_0) | [HuggingFace](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF) |
| Qwen3-ASR-0.6B | ~0.6 GB (Q8_0) | [HuggingFace](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF) |

Each model requires a main GGUF file and a corresponding mmproj GGUF file. Both are included in the download links above.

A **FireRedVAD** (Voice Activity Detection) model is bundled in the `models/` directory for speech segmentation.

## Dependencies

- `llama.cpp/` present in the project root (vendored; built from source).
- CLI11 (system-installed) for command-line parsing.
- GoogleTest (system-installed) for unit tests.

## Build

### Basic Build (CPU only)

```sh
cmake -S . -B build -DASR_BUILD_ENGINE=ON
cmake --build build -j
```

### GPU Acceleration

llama.cpp supports multiple GPU backends. Pass the corresponding CMake flag when building:

| Backend | CMake Flag | Platform | Notes |
| --- | --- | --- | --- |
| **CUDA** | `-DGGML_CUDA=ON` | Linux / Windows | Requires [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) |
| **ROCm (HIP)** | `-DGGML_HIP=ON` | Linux / Windows | Requires [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html). Optionally set `GPU_TARGETS` (e.g. `gfx1030`) |
| **Metal** | `-DGGML_METAL=ON` (default on macOS) | macOS | Uses Apple GPU; disable with `-DGGML_METAL=OFF` |
| **Vulkan** | `-DGGML_VULKAN=ON` | Linux / Windows / macOS | Requires Vulkan SDK |
| **BLAS** | `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` | Linux | Accelerates prompt processing with large batch sizes |
| **CPU** | _(default)_ | All | No extra flags needed |

#### Examples

**CUDA (NVIDIA GPU)**

```sh
cmake -S . -B build -DASR_BUILD_ENGINE=ON -DGGML_CUDA=ON
cmake --build build -j
```

**ROCm (AMD GPU, e.g. gfx1030)**

```sh
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build -DASR_BUILD_ENGINE=ON -DGGML_HIP=ON -DGPU_TARGETS=gfx1030
cmake --build build -j
```

**Metal (macOS)**

```sh
cmake -S . -B build -DASR_BUILD_ENGINE=ON
cmake --build build -j
```

**Vulkan**

```sh
cmake -S . -B build -DASR_BUILD_ENGINE=ON -DGGML_VULKAN=ON
cmake --build build -j
```

**OpenBLAS**

```sh
cmake -S . -B build -DASR_BUILD_ENGINE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build -j
```

### Core-only Build (no GPU, logic tests only)

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

## Audio Preprocessing

The tool accepts WAV, MP3, FLAC, and other formats supported by ffmpeg. For best results, convert audio to 16 kHz mono 16-bit PCM WAV beforehand:

```sh
ffmpeg -y -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

**Batch conversion** (all files in a directory):

```sh
for i in *.mp3; do
    wav_file="${i%.*}.wav"
    ffmpeg -y -i "$i" -ar 16000 -ac 1 -c:a pcm_s16le "$wav_file"
done
```

| Flag | Meaning |
| --- | --- |
| `-y` | Overwrite output without asking |
| `-ar 16000` | Resample to 16 kHz (required by the model) |
| `-ac 1` | Downmix to mono |
| `-c:a pcm_s16le` | 16-bit signed little-endian PCM codec |

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

The transcription always streams to `stdout`; file output flags additionally write files. Subtitle-like formats require VAD so timestamps are available.

## Testing

`ctest` runs the pure-logic suite by default. Engine and model-dependent tests are available in an `ASR_BUILD_ENGINE=ON` build; model-dependent cases are skipped unless `ASR_RUN_MODEL_TESTS=1` is set and the models are present:

```sh
ASR_RUN_MODEL_TESTS=1 ctest --test-dir build-engine
```

---

# 中文

基于 **llama.cpp / mtmd** 的独立语音识别工具，类似 `whisper-cli` 的使用方式。首个目标模型为 **Qwen3-ASR**；通过可扩展的模型配置（profile）机制，核心代码与具体模型解耦，未来可直接支持更多 mtmd 音频模型。

## 当前状态

已实现端到端文本转录，支持可选的 VAD 语音分段。输出格式包括 stdout 以及 txt/json/srt/vtt/lrc/csv 文件。长音频通过固定窗口分段或 FireRedVAD 语音活动检测进行处理。

## 支持模型

| 模型 | 大小 | 下载地址 |
| --- | --- | --- |
| Qwen3-ASR-1.7B | ~1.7 GB (Q8_0) | [HuggingFace](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF) |
| Qwen3-ASR-0.6B | ~0.6 GB (Q8_0) | [HuggingFace](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF) |

每个模型需要主模型 GGUF 文件和对应的 mmproj GGUF 文件，均可在上述链接下载。

**FireRedVAD**（语音活动检测）模型已放置在 `models/` 目录中，用于语音分段。

## 依赖

- `llama.cpp/` 位于项目根目录（以源码方式集成）
- CLI11（系统安装），用于命令行解析
- GoogleTest（系统安装），用于单元测试

## 编译

### 基本编译（仅 CPU）

```sh
cmake -S . -B build -DASR_BUILD_ENGINE=ON
cmake --build build -j
```

### GPU 加速

llama.cpp 支持多种 GPU 后端，编译时传入对应的 CMake 参数即可启用：

| 后端 | CMake 参数 | 平台 | 说明 |
| --- | --- | --- | --- |
| **CUDA** | `-DGGML_CUDA=ON` | Linux / Windows | 需安装 [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) |
| **ROCm (HIP)** | `-DGGML_HIP=ON` | Linux / Windows | 需安装 [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)，可设置 `GPU_TARGETS`（如 `gfx1030`） |
| **Metal** | `-DGGML_METAL=ON`（macOS 默认开启） | macOS | 使用 Apple GPU；用 `-DGGML_METAL=OFF` 关闭 |
| **Vulkan** | `-DGGML_VULKAN=ON` | Linux / Windows / macOS | 需安装 Vulkan SDK |
| **BLAS** | `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` | Linux | 加速大 batch 的 prompt 处理 |
| **CPU** | _（默认）_ | 全平台 | 无需额外参数 |

#### 示例

**CUDA（NVIDIA GPU）**

```sh
cmake -S . -B build -DASR_BUILD_ENGINE=ON -DGGML_CUDA=ON
cmake --build build -j
```

**ROCm（AMD GPU，如 gfx1030）**

```sh
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build -DASR_BUILD_ENGINE=ON -DGGML_HIP=ON -DGPU_TARGETS=gfx1030
cmake --build build -j
```

**Metal（macOS）**

```sh
cmake -S . -B build -DASR_BUILD_ENGINE=ON
cmake --build build -j
```

**Vulkan**

```sh
cmake -S . -B build -DASR_BUILD_ENGINE=ON -DGGML_VULKAN=ON
cmake --build build -j
```

**OpenBLAS**

```sh
cmake -S . -B build -DASR_BUILD_ENGINE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build -j
```

### 仅核心编译（无 GPU，仅逻辑测试）

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

## 音频预处理

工具支持 WAV、MP3、FLAC 等 ffmpeg 支持的格式。为获得最佳效果，建议提前将音频转为 16 kHz 单声道 16-bit PCM WAV：

```sh
ffmpeg -y -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

**批量转换**（目录下所有文件）：

```sh
for i in *.mp3; do
    wav_file="${i%.*}.wav"
    ffmpeg -y -i "$i" -ar 16000 -ac 1 -c:a pcm_s16le "$wav_file"
done
```

| 参数 | 含义 |
| --- | --- |
| `-y` | 覆盖输出文件，不询问 |
| `-ar 16000` | 重采样为 16 kHz（模型要求） |
| `-ac 1` | 混合为单声道 |
| `-c:a pcm_s16le` | 16-bit 有符号小端 PCM 编码 |

## 使用方法

```sh
./build/asr-cli \
    -m models/Qwen3-ASR-0.6B-Q8_0.gguf \
    --mmproj models/mmproj-Qwen3-ASR-0.6B-bf16.gguf \
    audio.wav -otxt -oj
```

常用选项：

| 选项 | 含义 |
| --- | --- |
| `-m, --model FNAME` | 主模型 GGUF 文件（必需） |
| `--mmproj FNAME` | 多模态投影 GGUF 文件（必需） |
| `-f, --file FNAME` | 输入音频（wav/mp3/flac）；也可作为位置参数 |
| `-of, --output-file BASE` | 输出基础路径（生成 `BASE.<格式>`） |
| `-otxt` / `-oj` | 输出 `.txt` / `.json` |
| `-osrt` / `-ovtt` | 输出 `.srt` / `.vtt` 字幕；需配合 `--vad` |
| `-olrc` / `-ocsv` | 输出 `.lrc` / `.csv`；需配合 `--vad` |
| `--output-format FMT` | 添加输出格式：`txt`、`json`、`srt`、`vtt`、`lrc`、`csv` |
| `--vad, --vad-model FNAME` | 使用 FireRedVAD 进行语音分段 |
| `--chunk-length N` | 分段长度（秒），默认 30 |
| `--context TEXT` | 传递给模型的热词/领域提示 |
| `--profile NAME` | 覆盖自动检测的模型配置 |
| `-t, --threads N` | CPU 线程数 |
| `-ng, --no-gpu` | 禁用 GPU |
| `-np, --no-prints` | 静默日志（转录结果仍输出到 stdout） |
| `-n, --n-predict N` | 每个分段的最大生成 token 数 |
| `-p, --processors N` | 并行推理实例数 |

转录结果始终输出到 `stdout`；文件输出选项会额外写入文件。字幕类格式需要 VAD 以获取时间戳。

## 测试

默认运行纯逻辑测试套件。引擎和模型相关测试在 `ASR_BUILD_ENGINE=ON` 构建中可用；模型相关测试需设置 `ASR_RUN_MODEL_TESTS=1` 且模型文件存在时才会执行：

```sh
ASR_RUN_MODEL_TESTS=1 ctest --test-dir build-engine
```
