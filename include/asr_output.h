#pragma once

#include "asr.h"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace asr {

// Escape a UTF-8 string for inclusion inside a JSON string literal. Multi-byte
// UTF-8 bytes (>= 0x80) are passed through verbatim (valid in JSON); only the
// mandatory escapes and control characters are encoded.
std::string json_escape(const std::string & s);

// Write the transcription as plain text: the full text followed by a newline.
void write_txt(std::ostream & os, const result & r);

// Write the transcription as minimal JSON: {"language": "...", "text": "..."}.
void write_json(std::ostream & os, const result & r);

// Write the transcription as full JSON with segments:
// {"language": "...", "text": "...", "segments": [{"start": ..., "end": ..., "text": "..."}]}
// Times are in seconds (float), matching whisper's JSON output format.
void write_json_full(std::ostream & os, const result & r);

// ---- Subtitles (SRT / VTT) ----

struct subtitle_cue {
    int         index    = 0;      // 1-based cue number
    int64_t     start_ms = 0;
    int64_t     end_ms   = 0;
    std::string text;
};

// Convert milliseconds to "HH:MM:SS,mmm" (SRT) or "HH:MM:SS.mmm" (VTT).
std::string to_srt_timestamp(int64_t ms);
std::string to_vtt_timestamp(int64_t ms);

// Write all cues to an SRT or VTT stream.
void write_srt(std::ostream & os, const std::vector<subtitle_cue> & cues);
void write_vtt(std::ostream & os, const std::vector<subtitle_cue> & cues);

// Split an asr::result into subtitle cues:
//   - Each segment becomes one or more cues, split on sentence-ending
//     punctuation (。！？.!?) when the text exceeds max_chars.
//   - Within a segment, sub-cue times are interpolated proportionally by
//     character count (CJK-aware: multi-byte UTF-8 counted as one codepoint).
//   - CJK punctuation 。！？ triggers a sentence-end split; 、， triggers a
//     clause-break split (used only when the text is already too long).
//   - CJK codepoints are counted without adding spurious spaces.
struct cue_params {
    int  max_chars     = 42; // max codepoints per cue (typical subtitle line)
    bool split_on_punct = true; // split on sentence-ending punctuation
};
std::vector<subtitle_cue> split_cues(const result & r, const cue_params & p = {});

} // namespace asr
