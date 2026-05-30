#pragma once

#include "asr.h"
#include "asr_engine.h"

#include <string>

namespace asr {

// Default chunk length (seconds) when transcribe_params.chunk_length_s <= 0.
// Tuned empirically in Phase 3; 30s is a safe starting point.
constexpr float kDefaultChunkLengthS = 30.0f;

// Transcribe one audio file end to end: load -> chunk -> per-chunk transcribe
// (printing each cleaned chunk to stdout unless quiet) -> merge -> result.
// Returns false if the audio could not be loaded.
bool transcribe_file(asr_context & ctx, const std::string & path,
                     const transcribe_params & tp, bool quiet, result & out);

} // namespace asr
