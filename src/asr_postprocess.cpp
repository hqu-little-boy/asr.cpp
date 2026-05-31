#include "asr_postprocess.h"
#include "asr_utf8.h"

#include <cctype>
#include <string>
#include <vector>

namespace asr {

namespace {

// Re-join codepoints into a single string.
std::string from_codepoints(const std::vector<std::string> & cps) {
    std::string out;
    for (const auto & c : cps) out += c;
    return out;
}

} // namespace

std::string fix_char_repeats(const std::string & s, int threshold) {
    const auto cps = utf8::to_codepoints(s);
    std::vector<std::string> out;
    std::string prev;
    int count = 0;
    for (const auto & c : cps) {
        if (c == prev) {
            ++count;
        } else {
            if (count > threshold) {
                // Replace the run with `threshold` copies.
                for (int i = 0; i < threshold; i++) out.push_back(prev);
            } else {
                for (int i = 0; i < count; i++) out.push_back(prev);
            }
            prev  = c;
            count = 1;
        }
    }
    // Flush last run.
    if (count > 0) {
        if (count > threshold) {
            for (int i = 0; i < threshold; i++) out.push_back(prev);
        } else {
            for (int i = 0; i < count; i++) out.push_back(prev);
        }
    }
    return from_codepoints(out);
}

std::string fix_pattern_repeats(const std::string & s, int threshold) {
    const auto cps = utf8::to_codepoints(s);
    if (cps.size() < 2) return s;

    // Try n-gram sizes 2..6.
    for (size_t n = 2; n <= 6 && n * 2 <= cps.size(); ++n) {
        std::vector<std::string> out;
        size_t i = 0;
        bool changed = false;
        while (i < cps.size()) {
            // Check if cps[i..i+n] repeats ≥ threshold times starting at i.
            if (i + n * 2 <= cps.size()) {
                int reps = 1;
                while (i + (reps + 1) * n <= cps.size()) {
                    bool match = true;
                    for (size_t k = 0; k < n; k++) {
                        if (cps[i + k] != cps[i + reps * n + k]) {
                            match = false;
                            break;
                        }
                    }
                    if (!match) break;
                    ++reps;
                }
                if (reps >= threshold) {
                    // Emit one copy of the pattern.
                    for (size_t k = 0; k < n; k++) out.push_back(cps[i + k]);
                    i += reps * n;
                    changed = true;
                    continue;
                }
            }
            out.push_back(cps[i]);
            ++i;
        }
        if (changed) {
            // Recurse to catch remaining patterns.
            return fix_pattern_repeats(from_codepoints(out), threshold);
        }
    }
    return s;
}

std::string suppress_repeats(const std::string & s, int threshold) {
    return fix_pattern_repeats(fix_char_repeats(s, threshold), threshold);
}

namespace {
// Compute the length (in codepoints) of the longest common prefix of a and b.
size_t common_prefix_cps(const std::string & a, const std::string & b) {
    size_t n = 0, ia = 0, ib = 0;
    while (ia < a.size() && ib < b.size()) {
        const size_t next_a = utf8::next_offset(a, ia);
        const size_t next_b = utf8::next_offset(b, ib);
        const size_t la = next_a - ia;
        const size_t lb = next_b - ib;
        if (la != lb || a.compare(ia, la, b, ib, lb) != 0) break;
        ++n;
        ia = next_a;
        ib = next_b;
    }
    return n;
}
} // namespace

void dedup_segments(result & r, size_t min_common) {
    if (r.segments.size() < 2) return;

    std::vector<segment> deduped;
    deduped.reserve(r.segments.size());
    deduped.push_back(r.segments[0]);

    for (size_t i = 1; i < r.segments.size(); ++i) {
        const std::string & prev = deduped.back().text;
        const std::string & cur  = r.segments[i].text;
        if (prev.empty() || cur.empty()) {
            if (!cur.empty()) deduped.push_back(r.segments[i]);
            continue;
        }
        const size_t cp_prev = utf8::count_codepoints(prev);
        const size_t cp_cur  = utf8::count_codepoints(cur);
        const size_t cp_com  = common_prefix_cps(prev, cur);

        // If one text is a prefix of the other (≥ min_common codepoints),
        // keep the longer one and drop the shorter.
        if (cp_com >= min_common) {
            if (cp_com >= cp_prev && cp_cur > cp_prev) {
                // prev is a prefix of cur → drop prev, keep cur.
                deduped.back() = r.segments[i];
            } else if (cp_com >= cp_cur && cp_prev > cp_cur) {
                // cur is a prefix of prev → drop cur (do nothing).
            } else {
                // Long common prefix but neither is a full prefix → keep both.
                deduped.push_back(r.segments[i]);
            }
        } else {
            deduped.push_back(r.segments[i]);
        }
    }

    // Rebuild the result.
    r.segments = std::move(deduped);
    r.text.clear();
    for (size_t i = 0; i < r.segments.size(); ++i) {
        if (i > 0 && !r.text.empty()) {
            const unsigned char back  = (unsigned char) r.text.back();
            const unsigned char front = (unsigned char) r.segments[i].text.front();
            if (back < 0x80 && std::isalnum(back) && front < 0x80 && std::isalnum(front)) {
                r.text += ' ';
            }
        }
        r.text += r.segments[i].text;
    }
}

} // namespace asr
