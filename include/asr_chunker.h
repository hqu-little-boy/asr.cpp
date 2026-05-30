#pragma once

#include <cstddef>
#include <vector>

namespace asr {

// A contiguous window of the input PCM, in sample units.
struct audio_chunk {
    size_t offset; // start sample
    size_t length; // number of samples
};

// Split a mono PCM signal of `n_samples` into windows of roughly
// `chunk_length_s` seconds each. To avoid cutting in the middle of a word, the
// boundary is nudged to the lowest-energy (quietest) point within +/- a search
// window around each target boundary -- but only when there is a real dip
// (energy well below the local mean); otherwise the cut stays at the target.
//
// Guarantees: windows are contiguous and cover the whole signal exactly
// (sum of lengths == n_samples, first offset == 0). chunk_length_s <= 0 or a
// signal shorter than one chunk yields a single window covering everything.
//
// This is pure logic (no model / no audio I/O) so it is fully unit-testable.
std::vector<audio_chunk> chunk_audio(const float * pcm,
                                     size_t        n_samples,
                                     int           sample_rate,
                                     float         chunk_length_s,
                                     float         search_s = 2.0f);

} // namespace asr
