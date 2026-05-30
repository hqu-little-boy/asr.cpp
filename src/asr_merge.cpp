#include "asr_merge.h"

#include <cctype>

namespace asr {

namespace {

// True for ASCII alphanumeric characters (UTF-8 continuation bytes, CJK, and
// punctuation are all false), used to decide whether to join two chunks with a
// space.
bool is_ascii_word(char c) {
    const unsigned char u = (unsigned char) c;
    return u < 0x80 && std::isalnum(u);
}

int64_t samples_to_ms(size_t samples, int sample_rate) {
    return (int64_t) samples * 1000 / sample_rate;
}

} // namespace

result merge_chunks(const std::vector<chunk_result> & chunks, int sample_rate) {
    result r;
    if (sample_rate <= 0) {
        sample_rate = 16000; // defensive; callers pass the model rate
    }

    for (const auto & c : chunks) {
        // Language: keep the first non-empty one, regardless of text content.
        if (r.language.empty() && !c.parsed.language.empty()) {
            r.language = c.parsed.language;
        }
        if (c.parsed.text.empty()) {
            continue;
        }

        segment seg;
        seg.t0_ms = samples_to_ms(c.span.offset, sample_rate);
        seg.t1_ms = samples_to_ms(c.span.offset + c.span.length, sample_rate);
        seg.text  = c.parsed.text;
        r.segments.push_back(seg);

        if (!r.text.empty() &&
            is_ascii_word(r.text.back()) && is_ascii_word(c.parsed.text.front())) {
            r.text += ' ';
        }
        r.text += c.parsed.text;
    }

    return r;
}

} // namespace asr
