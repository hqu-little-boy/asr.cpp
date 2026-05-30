#pragma once

#include "asr.h"

#include <ostream>
#include <string>

namespace asr {

// Escape a UTF-8 string for inclusion inside a JSON string literal. Multi-byte
// UTF-8 bytes (>= 0x80) are passed through verbatim (valid in JSON); only the
// mandatory escapes and control characters are encoded.
std::string json_escape(const std::string & s);

// Write the transcription as plain text: the full text followed by a newline.
void write_txt(std::ostream & os, const result & r);

// Write the transcription as minimal JSON: {"language": "...", "text": "..."}.
void write_json(std::ostream & os, const result & r);

} // namespace asr
