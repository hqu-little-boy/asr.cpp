#include "asr_driver.h"
#include "asr_chunker.h"
#include "asr_merge.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace asr {

bool transcribe_file(asr_context & ctx, const std::string & path,
                     const transcribe_params & tp, bool quiet, result & out) {
    std::vector<float> pcm;
    if (!ctx.load_audio(path, pcm)) {
        std::fprintf(stderr, "error: failed to load audio '%s'\n", path.c_str());
        return false;
    }

    const int   sr        = ctx.sample_rate();
    const float chunk_len = tp.chunk_length_s > 0.0f ? tp.chunk_length_s : kDefaultChunkLengthS;

    const std::vector<audio_chunk> windows = chunk_audio(pcm.data(), pcm.size(), sr, chunk_len);

    if (!quiet) {
        std::fprintf(stderr, "asr: '%s' — %.1f s, %zu chunk(s), profile '%s'\n",
                     path.c_str(), (double) pcm.size() / std::max(1, sr),
                     windows.size(), ctx.profile_name().c_str());
    }

    std::vector<chunk_result> results;
    results.reserve(windows.size());
    for (const auto & w : windows) {
        std::vector<float> sub(pcm.begin() + w.offset, pcm.begin() + w.offset + w.length);
        chunk_text ct = ctx.transcribe_chunk(sub, tp);
        if (!quiet && !ct.text.empty()) {
            std::printf("%s", ct.text.c_str());
            std::fflush(stdout);
        }
        results.push_back({w, std::move(ct)});
    }
    if (!quiet) {
        std::printf("\n");
        std::fflush(stdout);
    }

    out = merge_chunks(results, sr);
    return true;
}

} // namespace asr
