#include "asr_carry.h"

namespace asr {

namespace {
// Return the last <= max_chars bytes of s, advanced forward to the next UTF-8
// lead byte so the result never starts mid-codepoint.
std::string utf8_tail(const std::string & s, size_t max_chars) {
    if (s.size() <= max_chars) {
        return s;
    }
    size_t start = s.size() - max_chars;
    while (start < s.size() && ((unsigned char) s[start] & 0xC0) == 0x80) {
        ++start;
    }
    return s.substr(start);
}
} // namespace

std::string carry_context(const std::string & prev_text,
                          const std::string & base_context,
                          size_t              max_chars) {
    const std::string tail = utf8_tail(prev_text, max_chars);
    if (tail.empty()) {
        return base_context;
    }
    if (base_context.empty()) {
        return tail;
    }
    return base_context + " " + tail;
}

} // namespace asr
