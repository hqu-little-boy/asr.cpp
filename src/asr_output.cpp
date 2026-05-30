#include "asr_output.h"

#include <cstdio>

namespace asr {

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) c);
                    out += buf;
                } else {
                    out += (char) c; // printable ASCII or UTF-8 continuation byte
                }
        }
    }
    return out;
}

void write_txt(std::ostream & os, const result & r) {
    os << r.text << "\n";
}

void write_json(std::ostream & os, const result & r) {
    os << "{\n";
    os << "  \"language\": \"" << json_escape(r.language) << "\",\n";
    os << "  \"text\": \""     << json_escape(r.text)     << "\"\n";
    os << "}\n";
}

// ---- Subtitles ----

static std::string ts_to_string(int64_t ms, char sep) {
    // sep = ',' for SRT, '.' for VTT.
    const int64_t h  = ms / 3600000;
    const int64_t m  = (ms % 3600000) / 60000;
    const int64_t s  = (ms % 60000) / 1000;
    const int64_t msr = ms % 1000;
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld%c%03lld",
                  (long long) h, (long long) m, (long long) s, sep, (long long) msr);
    return buf;
}

std::string to_srt_timestamp(int64_t ms) { return ts_to_string(ms, ','); }
std::string to_vtt_timestamp(int64_t ms) { return ts_to_string(ms, '.'); }

void write_srt(std::ostream & os, const std::vector<subtitle_cue> & cues) {
    for (const auto & c : cues) {
        os << c.index << "\n"
           << to_srt_timestamp(c.start_ms) << " --> " << to_srt_timestamp(c.end_ms) << "\n"
           << c.text << "\n\n";
    }
}

void write_vtt(std::ostream & os, const std::vector<subtitle_cue> & cues) {
    os << "WEBVTT\n\n";
    for (const auto & c : cues) {
        os << to_vtt_timestamp(c.start_ms) << " --> " << to_vtt_timestamp(c.end_ms) << "\n"
           << c.text << "\n\n";
    }
}

namespace {

// Count UTF-8 codepoints (each lead byte or single-byte char = 1 codepoint).
size_t count_codepoints(const std::string & s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size(); ) {
        ++n;
        const unsigned char c = (unsigned char) s[i];
        i += (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
    }
    return n;
}

// True if byte is a UTF-8 lead byte (start of a multi-byte sequence).
bool is_lead(unsigned char c) { return (c & 0xC0) != 0x80; }

// Byte offset of the n-th codepoint (0-indexed) in s.
size_t codepoint_offset(const std::string & s, size_t n) {
    size_t cp = 0;
    for (size_t i = 0; i < s.size(); ) {
        if (cp == n) return i;
        ++cp;
        const unsigned char c = (unsigned char) s[i];
        i += (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
    }
    return s.size();
}

// Codepoint value at byte offset pos (for CJK punctuation detection).
uint32_t codepoint_at(const std::string & s, size_t pos) {
    if (pos >= s.size()) return 0;
    const unsigned char c = (unsigned char) s[pos];
    if (c < 0x80) return c;
    if (c < 0xE0) return ((c & 0x1F) << 6) | ((unsigned char) s[pos + 1] & 0x3F);
    if (c < 0xF0) return ((c & 0x0F) << 12) | (((unsigned char) s[pos + 1] & 0x3F) << 6) |
                          ((unsigned char) s[pos + 2] & 0x3F);
    return ((c & 0x07) << 18) | (((unsigned char) s[pos + 1] & 0x3F) << 12) |
           (((unsigned char) s[pos + 2] & 0x3F) << 6) | ((unsigned char) s[pos + 3] & 0x3F);
}

bool is_sentence_end(uint32_t cp) {
    return cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F ||  // 。！？
           cp == '.'    || cp == '!'   || cp == '?';
}

bool is_clause_break(uint32_t cp) {
    return cp == 0x3001 || cp == 0xFF0C ||  // 、，
           cp == ','    || cp == ';';
}

// Split text into pieces at positions where is_break(codepoint) is true.
// Each piece is a {byte_start, byte_end} range into `text`.
struct span { size_t start, end; };
std::vector<span> split_at_breaks(const std::string & text,
                                  bool (*pred)(uint32_t),
                                  size_t min_codepoints_before) {
    std::vector<span> pieces;
    size_t piece_start = 0;
    size_t cp_count    = 0;
    for (size_t i = 0; i < text.size(); ) {
        const size_t next = i + ((unsigned char) text[i] < 0x80 ? 1 :
                                 (unsigned char) text[i] < 0xE0 ? 2 :
                                 (unsigned char) text[i] < 0xF0 ? 3 : 4);
        ++cp_count;
        if (pred(codepoint_at(text, i)) && cp_count >= min_codepoints_before) {
            pieces.push_back({piece_start, next});
            piece_start = next;
            cp_count    = 0;
        }
        i = next;
    }
    if (piece_start < text.size()) {
        pieces.push_back({piece_start, text.size()});
    }
    return pieces;
}

// Interpolate times for a sub-cue based on its byte range within the full text.
int64_t interp_time(int64_t t0, int64_t t1, size_t total_bytes, size_t byte_pos) {
    if (total_bytes == 0) return t0;
    return t0 + (int64_t)((double)(t1 - t0) * byte_pos / total_bytes);
}

// Emit cues for a single segment's text, splitting as needed.
void emit_cues_for_segment(const std::string & text, int64_t t0, int64_t t1,
                           const cue_params & p, int & cue_index,
                           std::vector<subtitle_cue> & out) {
    const size_t n_cp = count_codepoints(text);
    if (n_cp == 0) return;

    if (n_cp <= (size_t) p.max_chars || !p.split_on_punct) {
        subtitle_cue c;
        c.index    = cue_index++;
        c.start_ms = t0;
        c.end_ms   = t1;
        c.text     = text;
        out.push_back(c);
        return;
    }

    // Try splitting at sentence-ending punctuation first (min 2 codepoints
    // before a break so very short fragments like "。" alone are avoided).
    auto pieces = split_at_breaks(text, is_sentence_end, 2);
    if (pieces.size() <= 1) {
        // No sentence breaks found; try clause breaks.
        pieces = split_at_breaks(text, is_clause_break, 2);
    }
    if (pieces.size() <= 1) {
        // Still no breaks; force-split at max_chars codepoints.
        pieces.clear();
        for (size_t cp = 0; cp < n_cp; ) {
            const size_t end_cp = std::min(cp + (size_t) p.max_chars, n_cp);
            pieces.push_back({codepoint_offset(text, cp), codepoint_offset(text, end_cp)});
            cp = end_cp;
        }
    }

    // Emit one cue per piece, interpolating times proportionally by byte position.
    for (const auto & sp : pieces) {
        const std::string piece = text.substr(sp.start, sp.end - sp.start);
        if (piece.empty()) continue;
        subtitle_cue c;
        c.index    = cue_index++;
        c.start_ms = interp_time(t0, t1, text.size(), sp.start);
        c.end_ms   = interp_time(t0, t1, text.size(), sp.end);
        c.text     = piece;
        out.push_back(c);
    }
}

} // namespace

std::vector<subtitle_cue> split_cues(const result & r, const cue_params & p) {
    std::vector<subtitle_cue> cues;
    int cue_index = 1;
    for (const auto & seg : r.segments) {
        emit_cues_for_segment(seg.text, seg.t0_ms, seg.t1_ms, p, cue_index, cues);
    }
    return cues;
}

} // namespace asr
