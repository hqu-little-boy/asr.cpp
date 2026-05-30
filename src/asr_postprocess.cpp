#include "asr_postprocess.h"

#include <string>
#include <vector>

namespace asr {

namespace {

// Decode one UTF-8 codepoint starting at `pos`, advance `pos`, return the
// codepoint as a std::string (the raw UTF-8 bytes). Returns "" if at end.
std::string next_cp(const std::string & s, size_t & pos) {
    if (pos >= s.size()) return {};
    const unsigned char c = (unsigned char) s[pos];
    const size_t len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
    const size_t end = std::min(pos + len, s.size());
    std::string cp(s, pos, end - pos);
    pos = end;
    return cp;
}

// Decode the full string into a vector of UTF-8 codepoint strings.
std::vector<std::string> to_codepoints(const std::string & s) {
    std::vector<std::string> cps;
    size_t pos = 0;
    while (pos < s.size()) cps.push_back(next_cp(s, pos));
    return cps;
}

// Re-join codepoints into a single string.
std::string from_codepoints(const std::vector<std::string> & cps) {
    std::string out;
    for (const auto & c : cps) out += c;
    return out;
}

} // namespace

std::string fix_char_repeats(const std::string & s, int threshold) {
    const auto cps = to_codepoints(s);
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
    const auto cps = to_codepoints(s);
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

} // namespace asr
