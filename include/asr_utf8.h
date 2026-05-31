#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace asr::utf8 {

inline bool is_continuation(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

inline size_t codepoint_length(unsigned char c) {
    return (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
}

inline size_t next_offset(std::string_view s, size_t pos) {
    if (pos >= s.size()) {
        return s.size();
    }
    return std::min(pos + codepoint_length((unsigned char) s[pos]), s.size());
}

inline size_t count_codepoints(std::string_view s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size(); i = next_offset(s, i)) {
        ++n;
    }
    return n;
}

inline size_t codepoint_offset(std::string_view s, size_t n) {
    size_t cp = 0;
    for (size_t i = 0; i < s.size(); i = next_offset(s, i), ++cp) {
        if (cp == n) {
            return i;
        }
    }
    return s.size();
}

inline uint32_t codepoint_at(std::string_view s, size_t pos) {
    if (pos >= s.size()) return 0;
    const unsigned char c = (unsigned char) s[pos];
    if (c < 0x80) return c;
    if (pos + 1 >= s.size()) return 0;
    if (c < 0xE0) return ((c & 0x1F) << 6) | ((unsigned char) s[pos + 1] & 0x3F);
    if (pos + 2 >= s.size()) return 0;
    if (c < 0xF0) return ((c & 0x0F) << 12) | (((unsigned char) s[pos + 1] & 0x3F) << 6) |
                          ((unsigned char) s[pos + 2] & 0x3F);
    if (pos + 3 >= s.size()) return 0;
    return ((c & 0x07) << 18) | (((unsigned char) s[pos + 1] & 0x3F) << 12) |
           (((unsigned char) s[pos + 2] & 0x3F) << 6) | ((unsigned char) s[pos + 3] & 0x3F);
}

inline std::vector<std::string> to_codepoints(std::string_view s) {
    std::vector<std::string> cps;
    for (size_t pos = 0; pos < s.size(); ) {
        const size_t next = next_offset(s, pos);
        cps.emplace_back(s.substr(pos, next - pos));
        pos = next;
    }
    return cps;
}

inline std::string tail(std::string_view s, size_t max_bytes) {
    if (s.size() <= max_bytes) {
        return std::string(s);
    }
    size_t start = s.size() - max_bytes;
    while (start < s.size() && is_continuation((unsigned char) s[start])) {
        ++start;
    }
    return std::string(s.substr(start));
}

} // namespace asr::utf8
