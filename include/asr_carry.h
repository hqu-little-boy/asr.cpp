#pragma once

#include <cstddef>
#include <string>

namespace asr {

// Build the context string for the next chunk when carry-over mode is enabled:
// the base context (hotwords) followed by a UTF-8-safe tail (at most max_chars
// bytes, not splitting a codepoint) of the accumulated prior transcript. If the
// prior transcript is empty, returns base_context unchanged. Pure logic.
std::string carry_context(const std::string & prev_text,
                          const std::string & base_context,
                          size_t              max_chars = 200);

} // namespace asr
