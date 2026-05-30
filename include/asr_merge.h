#pragma once

#include "asr.h"
#include "asr_chunker.h"

#include <vector>

namespace asr {

// One chunk's transcription paired with its position in the source audio.
struct chunk_result {
    audio_chunk span;   // location in the source PCM (samples)
    chunk_text  parsed; // {text, language} from the profile's parse_output
};

// Merge per-chunk results into a final transcription:
//   - text: non-empty chunk texts joined. A single space is inserted between
//     two chunks only when the adjacent characters are both ASCII word
//     characters (so English words stay separated while CJK is not split by
//     spurious spaces).
//   - segments: one per non-empty chunk, with t0_ms/t1_ms derived from the
//     chunk span and sample_rate.
//   - language: the first non-empty chunk language.
//
// Pure logic (no model / no audio I/O), fully unit-testable.
result merge_chunks(const std::vector<chunk_result> & chunks, int sample_rate);

} // namespace asr
