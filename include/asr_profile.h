#pragma once

#include "asr.h"

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

} // namespace asr
