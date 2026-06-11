#include "asr_driver.h"
#include "asr_chunker.h"
#include "asr_carry.h"
#include "asr_merge.h"
#include "asr_postprocess.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace asr {

namespace {

// Transcribe a single chunk (called from both sequential and parallel paths).
chunk_result transcribe_one(asr_context & ctx, const audio_chunk & w,
                            const std::vector<float> & pcm, int sr,
                            const transcribe_params & tp_chunk, bool quiet) {
    const std::span<const float> samples(pcm);
    chunk_text ct = ctx.transcribe_chunk(samples.subspan(w.offset, w.length), tp_chunk);
    if (!quiet && !ct.text.empty()) {
        std::printf("%s", ct.text.c_str());
        std::fflush(stdout);
    }
    return {w, std::move(ct)};
}

} // namespace

bool transcribe_file(asr_context & ctx, const std::string & path,
                     const transcribe_params & tp, bool quiet, result & out,
                     vad_context * vad, const vad_params & vp, int processors,
                     std::atomic<bool> * cancel) {
    std::vector<float> pcm;
    if (!ctx.load_audio(path, pcm)) {
        std::fprintf(stderr, "error: failed to load audio '%s'\n", path.c_str());
        return false;
    }

    const int sr = ctx.sample_rate();

    // Segment the audio: VAD or fixed-window chunker.
    std::vector<audio_chunk> windows;
    if (vad != nullptr) {
        const auto segs = vad->detect(pcm.data(), (int) pcm.size(), vp, cancel);
        for (const auto & s : segs) {
            audio_chunk w;
            w.offset = (size_t) (s.start_sec * sr);
            w.length = (size_t) ((s.end_sec - s.start_sec) * sr);
            if (w.offset + w.length > pcm.size()) w.length = pcm.size() - w.offset;
            if (w.length > 0) windows.push_back(w);
        }
    } else {
        const float chunk_len = tp.chunk_length_s > 0.0f ? tp.chunk_length_s : kDefaultChunkLengthS;
        windows = chunk_audio(std::span<const float>(pcm), sr, chunk_len);
    }

    if (!quiet) {
        std::fprintf(stderr, "asr: '%s' — %.1f s, %zu segment(s), profile '%s'%s\n",
                     path.c_str(), (double) pcm.size() / std::max(1, sr),
                     windows.size(), ctx.profile_name().c_str(),
                     vad ? ", vad" : "");
    }

    if (windows.empty()) {
        out = result{};
        return true;
    }

    const int n_workers = std::max(1, std::min(processors, (int) windows.size()));

    // Sequential path (n_workers == 1): process chunks in order.
    if (n_workers == 1) {
        std::vector<chunk_result> results;
        results.reserve(windows.size());
        std::string carry;
        for (size_t wi = 0; wi < windows.size(); ++wi) {
            if (cancel && cancel->load()) {
                std::fprintf(stderr, "\nasr: cancelled\n");
                break;
            }
            const auto & w = windows[wi];
            if (!quiet) {
                std::fprintf(stderr, "\rasr: [%zu/%zu] segment %.1f-%.1f s",
                             wi + 1, windows.size(),
                             (double) w.offset / sr,
                             (double) (w.offset + w.length) / sr);
                std::fflush(stderr);
            }
            transcribe_params tp_chunk = tp;
            if (tp.carry_context) {
                tp_chunk.context = carry_context(carry, tp.context);
            }
            auto cr = transcribe_one(ctx, w, pcm, sr, tp_chunk, quiet);
            if (tp.carry_context && !cr.parsed.text.empty()) {
                if (!carry.empty()) carry += ' ';
                carry += cr.parsed.text;
            }
            results.push_back(std::move(cr));
        }
        if (!quiet) std::fprintf(stderr, "\n");
        out = merge_chunks(results, sr);
        dedup_segments(out);
        return true;
    }

    // Parallel path (n_workers > 1): create N-1 worker contexts and distribute
    // chunks round-robin across workers. Each worker processes its chunks
    // independently (no carry-over in parallel mode).
    if (!quiet) {
        std::fprintf(stderr, "asr: creating %d parallel workers...\n", n_workers);
    }

    // Worker 0 = main context (not owned). Workers 1..N-1 = clones (owned).
    std::vector<asr_context *> workers;
    std::vector<std::unique_ptr<asr_context>> clones;
    workers.push_back(&ctx); // worker 0
    for (int i = 1; i < n_workers; ++i) {
        auto w = ctx.clone();
        if (!w) {
            std::fprintf(stderr, "error: failed to create worker %d, falling back to sequential\n", i);
            return transcribe_file(ctx, path, tp, quiet, out, vad, vp, 1);
        }
        workers.push_back(w.get());
        clones.push_back(std::move(w));
    }

    // Distribute chunks round-robin across workers.
    // Each worker gets a list of (chunk_index, chunk) pairs.
    struct work_item { size_t idx; audio_chunk chunk; };
    std::vector<std::vector<work_item>> per_worker(n_workers);
    for (size_t i = 0; i < windows.size(); ++i) {
        per_worker[i % n_workers].push_back({i, windows[i]});
    }

    // Per-worker results (indexed by chunk index).
    std::vector<chunk_result> all_results(windows.size());
    std::mutex print_mtx;

    auto worker_fn = [&](int worker_id) {
        asr_context & wctx = *workers[worker_id];
        for (const auto & item : per_worker[worker_id]) {
            if (cancel && cancel->load()) break;
            transcribe_params tp_chunk = tp;
            tp_chunk.carry_context = false; // no carry-over in parallel mode
            auto cr = transcribe_one(wctx, item.chunk, pcm, sr, tp_chunk, true); // quiet=true, print below
            all_results[item.idx] = std::move(cr);
            if (!quiet && !all_results[item.idx].parsed.text.empty()) {
                std::lock_guard<std::mutex> lk(print_mtx);
                std::printf("%s", all_results[item.idx].parsed.text.c_str());
                std::fflush(stdout);
            }
        }
    };

    // Launch workers (skip worker 0 which runs on the main thread).
    std::vector<std::thread> threads;
    for (int i = 1; i < n_workers; ++i) {
        threads.emplace_back(worker_fn, i);
    }
    worker_fn(0); // main thread = worker 0
    for (auto & t : threads) t.join();

    if (!quiet) std::fprintf(stderr, "\n");

    // Merge results in chunk order (all_results is already ordered by chunk index).
    std::vector<chunk_result> ordered;
    ordered.reserve(windows.size());
    for (auto & r : all_results) {
        if (r.parsed.text.empty() && r.span.length == 0) continue; // skip empty
        ordered.push_back(std::move(r));
    }
    out = merge_chunks(ordered, sr);
    dedup_segments(out);

    return true; // clones are freed when `clones` goes out of scope
}

} // namespace asr
