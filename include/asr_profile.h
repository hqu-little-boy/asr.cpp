#pragma once

#include "asr.h"

#include <functional>
#include <string>

namespace asr {

// Parse a Qwen3-ASR raw generation into {language, text}.
//
// Qwen3-ASR emits a protocol of the form:
//     language <Lang><asr_text><transcription>
// where "<asr_text>" is a special token that detokenizes to that literal
// string. This parser is pure string logic:
//   - split at the FIRST "<asr_text>" marker; everything after it is the
//     transcription (trimmed),
//   - the preamble before the marker is mined for a leading
//     "language <value>" (keyword matched case-insensitively; value is the
//     trimmed remainder), and is otherwise discarded,
//   - if no marker is present, the whole input is treated as the transcription
//     and language is empty (also the behaviour of the generic profile).
chunk_text parse_qwen3a_output(const std::string & raw);

// A per-model profile: the single extension point for new mtmd ASR models.
//   - build_prompt produces the user-message content for one chunk: the media
//     marker (supplied by the engine via mtmd_default_marker(), so this stays
//     model-agnostic and free of any mtmd dependency) followed by the context.
//   - parse_output turns the model's raw generation into {text, language}.
// Adding a new model == registering a new profile; no core changes required.
struct profile {
    std::string name; // matches the mmproj projector_type, or "generic"
    std::function<std::string(const transcribe_params & params,
                              const std::string &        media_marker)> build_prompt;
    std::function<chunk_text(const std::string & raw)>                  parse_output;
};

// Select a profile by the mmproj projector_type (e.g. "qwen3a"). Any
// unregistered type falls back to a generic pass-through profile.
const profile & select_profile(const std::string & projector_type);

} // namespace asr
