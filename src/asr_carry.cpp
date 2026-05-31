#include "asr_carry.h"
#include "asr_utf8.h"

namespace asr {

std::string carry_context(const std::string & prev_text,
                          const std::string & base_context,
                          size_t              max_chars) {
    const std::string tail = utf8::tail(prev_text, max_chars);
    if (tail.empty()) {
        return base_context;
    }
    if (base_context.empty()) {
        return tail;
    }
    return base_context + " " + tail;
}

} // namespace asr
