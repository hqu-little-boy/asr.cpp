#include "asr_chunker.h"

#include <algorithm>
#include <cmath>
#include <span>

namespace asr {

std::vector<audio_chunk> chunk_audio(std::span<const float> pcm,
                                     int                    sample_rate,
                                     float                  chunk_length_s,
                                     float                  search_s) {
    std::vector<audio_chunk> chunks;
    if (pcm.empty()) {
        return chunks;
    }
    if (sample_rate <= 0 || chunk_length_s <= 0.0f) {
        chunks.push_back({0, pcm.size()});
        return chunks;
    }

    const size_t target = (size_t) std::llround((double) chunk_length_s * sample_rate);
    if (target == 0 || pcm.size() <= target) {
        chunks.push_back({0, pcm.size()});
        return chunks;
    }

    const size_t frame  = std::max<size_t>(1, (size_t) (sample_rate / 100)); // ~10 ms
    const size_t search = (size_t) std::llround((double) std::max(0.0f, search_s) * sample_rate);

    size_t cur = 0;
    while (cur < pcm.size()) {
        const size_t ideal_end = cur + target;
        if (ideal_end >= pcm.size()) {
            chunks.push_back({cur, pcm.size() - cur});
            break;
        }

        // Search window around the ideal boundary, kept strictly ahead of `cur`.
        size_t lo = (ideal_end > search) ? ideal_end - search : 0;
        lo = std::max(lo, cur + frame);
        size_t hi = std::min(ideal_end + search, pcm.size());

        size_t cut = ideal_end;
        if (lo + frame <= hi) {
            double min_e = -1.0;
            double sum_e = 0.0;
            size_t cnt   = 0;
            size_t min_f = ideal_end;
            for (size_t f = lo; f + frame <= hi; f += frame) {
                double e = 0.0;
                for (size_t i = f; i < f + frame; ++i) {
                    e += (double) pcm[i] * (double) pcm[i];
                }
                sum_e += e;
                ++cnt;
                if (min_e < 0.0 || e < min_e) {
                    min_e = e;
                    min_f = f;
                }
            }
            // Only move the cut to a real silence dip (energy well below local mean).
            if (cnt > 0 && min_e < 0.3 * (sum_e / (double) cnt)) {
                cut = min_f + frame / 2;
            }
        }

        // Ensure forward progress and bounds.
        if (cut <= cur)        cut = ideal_end;
        if (cut > pcm.size())  cut = pcm.size();

        chunks.push_back({cur, cut - cur});
        cur = cut;
    }

    return chunks;
}

std::vector<audio_chunk> chunk_audio(const float * pcm,
                                     size_t        n_samples,
                                     int           sample_rate,
                                     float         chunk_length_s,
                                     float         search_s) {
    if (pcm == nullptr && n_samples > 0) {
        return {};
    }
    return chunk_audio(std::span<const float>(pcm, n_samples), sample_rate, chunk_length_s, search_s);
}

} // namespace asr
