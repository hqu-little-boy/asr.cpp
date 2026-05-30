#include "asr_driver.h"
#include "asr_chunker.h"
#include "asr_carry.h"
#include "asr_merge.h"
#include "asr_postprocess.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace asr {

bool transcribe_file(asr_context & ctx, const std::string & path,
                     const transcribe_params & tp, bool quiet, result & out,
                     vad_context * vad, const vad_params & vp) {
    std::vector<float> pcm;
    if (!ctx.load_audio(path, pcm)) {
        std::fprintf(stderr, "error: failed to load audio '%s'\n", path.c_str());
        return false;
    }

    const int sr = ctx.sample_rate();

    // Segment the audio: VAD or fixed-window chunker.
    std::vector<audio_chunk> windows;
    if (vad != nullptr) {
        const auto segs = vad->detect(pcm.data(), (int) pcm.size(), vp);
        for (const auto & s : segs) {
            audio_chunk w;
            w.offset = (size_t) (s.start_sec * sr);
            w.length = (size_t) ((s.end_sec - s.start_sec) * sr);
            if (w.offset + w.length > pcm.size()) w.length = pcm.size() - w.offset;
            if (w.length > 0) windows.push_back(w);
        }
    } else {
        const float chunk_len = tp.chunk_length_s > 0.0f ? tp.chunk_length_s : kDefaultChunkLengthS;
        windows = chunk_audio(pcm.data(), pcm.size(), sr, chunk_len);
    }

    if (!quiet) {
        std::fprintf(stderr, "asr: '%s' — %.1f s, %zu segment(s), profile '%s'%s\n",
                     path.c_str(), (double) pcm.size() / std::max(1, sr),
                     windows.size(), ctx.profile_name().c_str(),
                     vad ? ", vad" : "");
    }

    std::vector<chunk_result> results;
    results.reserve(windows.size());
    std::string carry; // accumulated transcript, used only in carry-over mode
    for (size_t wi = 0; wi < windows.size(); ++wi) {
        const auto & w = windows[wi];
        std::vector<float> sub(pcm.begin() + w.offset, pcm.begin() + w.offset + w.length);

        transcribe_params tp_chunk = tp;
        if (tp.carry_context) {
            tp_chunk.context = carry_context(carry, tp.context);
        }

        if (!quiet) {
            std::fprintf(stderr, "\rasr: [%zu/%zu] segment %.1f-%.1f s",
                         wi + 1, windows.size(),
                         (double) w.offset / sr,
                         (double) (w.offset + w.length) / sr);
            std::fflush(stderr);
        }

        chunk_text ct = ctx.transcribe_chunk(sub, tp_chunk);
        if (!ct.text.empty()) {
            std::printf("%s", ct.text.c_str());
            std::fflush(stdout);
        } else if (!quiet) {
            std::fprintf(stderr, "\rasr: [%zu/%zu] segment %.1f-%.1f s (empty)",
                         wi + 1, windows.size(),
                         (double) w.offset / sr,
                         (double) (w.offset + w.length) / sr);
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
    // Remove inter-segment duplicates (common when adjacent VAD segments
    // share boundary text).
    dedup_segments(out);
    return true;
}

} // namespace asr
