#include "asr_merge.h"

#include <cctype>
#include <span>
#include <string>
#include <string_view>

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

// Find the longest suffix of `prev` that is also a prefix of `next`, with
// length >= min_overlap. Returns the overlap length (0 if none found).
// Used to detect and remove duplicate text at chunk boundaries.
size_t find_overlap(std::string_view prev, std::string_view next, size_t min_overlap = 8) {
    const size_t max_len = std::min(prev.size(), next.size());
    for (size_t len = max_len; len >= min_overlap; --len) {
        if (prev.substr(prev.size() - len, len) == next.substr(0, len)) {
            return len;
        }
    }
    return 0;
}

} // namespace

namespace {

result merge_chunk_span(std::span<const chunk_result> chunks, int sample_rate) {
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

        // Dedup: if the tail of the previous segment's text matches the head
        // of this segment's text, trim the duplicate from this segment.
        if (!r.segments.empty()) {
            const std::string & prev_text = r.segments.back().text;
            const size_t overlap = find_overlap(prev_text, seg.text);
            if (overlap > 0) {
                // Trim the duplicate prefix from this segment.
                size_t trim = overlap;
                // Also skip any whitespace after the trimmed portion.
                while (trim < seg.text.size() && std::isspace((unsigned char) seg.text[trim])) {
                    ++trim;
                }
                seg.text = seg.text.substr(trim);
                if (seg.text.empty()) {
                    continue; // entire segment was a duplicate
                }
            }
        }

        r.segments.push_back(seg);

        if (!r.text.empty() &&
            is_ascii_word(r.text.back()) && is_ascii_word(seg.text.front())) {
            r.text += ' ';
        }
        r.text += seg.text;
    }

    return r;
}

} // namespace

result merge_chunks(const std::vector<chunk_result> & chunks, int sample_rate) {
    return merge_chunk_span(std::span<const chunk_result>(chunks), sample_rate);
}

} // namespace asr
