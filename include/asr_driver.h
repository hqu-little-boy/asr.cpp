#pragma once

#include "asr.h"
#include "asr_engine.h"
#include "asr_vad.h"

#include <string>

namespace asr {

// Default chunk length (seconds) when transcribe_params.chunk_length_s <= 0.
// Used only in the non-VAD (fixed-window) path.
constexpr float kDefaultChunkLengthS = 30.0f;

// Transcribe one audio file end to end:
//   - If vad is non-null, use VAD speech segments (with vad_params) as the
//     segmentation; each VAD segment is transcribed as an independent chunk.
//   - Otherwise, fall back to the fixed-window + energy-valley chunker.
//   - If processors > 1, chunks are distributed across N parallel inference
//     contexts (each with its own llama_context + mtmd context, sharing the
//     model). Results are merged in chunk order.
//
// Prints each cleaned chunk to stdout unless quiet. Merges all chunks into
// `out`. Returns false if the audio could not be loaded.
bool transcribe_file(asr_context & ctx, const std::string & path,
                     const transcribe_params & tp, bool quiet,
                     result & out,
                     vad_context * vad = nullptr,
                     const vad_params & vp = vad_params{},
                     int processors = 1);

} // namespace asr
