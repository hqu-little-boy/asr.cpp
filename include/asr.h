#pragma once

// asr.cpp public types and library entry points.
//
// Layering note:
//   - This header and the `asr_core` library are PURE: no llama.cpp / mtmd
//     dependency, so the bulk of the logic (parsing, chunking, formatting,
//     argument parsing, profile registry, merging) is unit-testable without a
//     model.
//   - The mtmd-backed engine lives in the separate `asr_engine` library and is
//     only built when ASR_BUILD_ENGINE is enabled.

#include <cstdint>
#include <string>
#include <vector>

namespace asr {

// Library version string.
const char * version();

// A contiguous span of transcribed text with (coarse) timing in milliseconds.
// In v1 timestamps are chunk-level only and not surfaced in the output.
struct segment {
    int64_t     t0_ms = 0;
    int64_t     t1_ms = 0;
    std::string text;
};

// Full transcription result, engine-agnostic. Consumed by the output layer.
struct result {
    std::string          text;     // full transcription (canonical output)
    std::vector<segment> segments; // structural pieces; timing for future use
    std::string          language; // detected language (may be empty)
};

// Parsed output of a single chunk's generation.
struct chunk_text {
    std::string text;
    std::string language;
};

// Parameters needed to load a model.
struct model_params {
    std::string model;                  // main GGUF model path (required)
    std::string mmproj;                 // multimodal projector GGUF path (required)
    bool        use_gpu        = true;
    bool        mmproj_use_gpu = true;
    int         n_threads      = 4;
    std::string profile_override;       // --profile; empty => auto-detect
};

// Parameters for a transcription run.
struct transcribe_params {
    std::string context;                // --context: hotwords / domain bias
    std::string language;               // --language: force language (e.g. "Chinese", "English")
    int         n_predict      = -1;    // per-chunk token cap; -1 => derive
    float       chunk_length_s = 0.0f;  // 0 => use tuned default
    bool        carry_context  = false; // feed prior transcript tail as context (experimental)
    float       temperature    = -1.0f; // sampling temperature; <0 = use default
    float       top_p          = -1.0f; // top-p sampling; <0 = use default
    float       repeat_penalty = -1.0f; // repeat penalty; <0 = use default
};

// Output selection (CLI-level).
struct output_params {
    std::string out_base;               // -of: output file base path
    bool        out_txt   = false;      // -otxt
    bool        out_json  = false;      // -oj
    bool        out_srt   = false;      // -osrt
    bool        out_vtt   = false;      // -ovtt
    bool        no_prints = false;      // -np
};

// VAD-related parameters (FireRedVAD).
struct vad_run_params {
    std::string model_path;             // --vad-model FNAME
    bool        use_vad  = false;       // --vad
    float       threshold       = 0.5f;
    float       min_speech_sec  = 0.25f;
    float       min_silence_sec = 0.10f;
};

} // namespace asr
