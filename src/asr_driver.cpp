#include "asr_driver.h"
#include "asr_chunker.h"
#include "asr_carry.h"
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
    std::string carry; // accumulated transcript, used only in carry-over mode
    for (const auto & w : windows) {
        std::vector<float> sub(pcm.begin() + w.offset, pcm.begin() + w.offset + w.length);

        transcribe_params tp_chunk = tp;
        if (tp.carry_context) {
            tp_chunk.context = carry_context(carry, tp.context);
        }

        chunk_text ct = ctx.transcribe_chunk(sub, tp_chunk);
        // The transcription is the primary result: always stream it to stdout
        // (so -np silences logs but still emits the text). Files are written by
        // the caller from the merged result.
        if (!ct.text.empty()) {
            std::printf("%s", ct.text.c_str());
            std::fflush(stdout);
        }
        if (tp.carry_context && !ct.text.empty()) {
            if (!carry.empty()) carry += ' ';
            carry += ct.text;
        }
        results.push_back({w, std::move(ct)});
    }
    std::printf("\n");
    std::fflush(stdout);

    out = merge_chunks(results, sr);
    return true;
}

} // namespace asr
