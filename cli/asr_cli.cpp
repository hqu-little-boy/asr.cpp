#include "asr_args.h"

#include <cstdio>
#include <csignal>

#ifdef ASR_WITH_ENGINE
#include "asr_engine.h"
#include "asr_driver.h"
#include "asr_output.h"
#include "asr_postprocess.h"
#include "asr_vad.h"

#include <atomic>
#include <fstream>
#include <string>

static std::atomic<bool> g_interrupted{false};

static void sigint_handler(int) {
    g_interrupted.store(true);
}
#endif

int main(int argc, char ** argv) {
    const asr::cli_args args = asr::parse_args(argc, argv);

    if (args.help) {
        std::fputs(asr::usage_string(argv[0]).c_str(), stdout);
        return 0;
    }
    if (args.error) {
        std::fprintf(stderr, "error: %s\n\n", args.error_msg.c_str());
        std::fputs(asr::usage_string(argv[0]).c_str(), stderr);
        return 1;
    }

#ifdef ASR_WITH_ENGINE
    // Register SIGINT handler for graceful stop.
    std::signal(SIGINT, sigint_handler);

    auto ctx = asr::asr_context::load(args.model, args.output.no_prints, args.transcribe);
    if (!ctx) {
        std::fprintf(stderr, "error: failed to load model / mmproj\n");
        return 1;
    }

    // Optionally load FireRedVAD.
    std::unique_ptr<asr::vad_context> vad;
    if (args.vad.use_vad) {
        vad = asr::vad_context::load(args.vad.model_path);
        if (!vad) {
            std::fprintf(stderr, "error: failed to load VAD model '%s'\n", args.vad.model_path.c_str());
            return 1;
        }
    }

    asr::vad_params vp;
    vp.threshold       = args.vad.threshold;
    vp.min_speech_sec  = args.vad.min_speech_sec;
    vp.min_silence_sec = args.vad.min_silence_sec;

    int ret = 0;
    for (const auto & file : args.input_files) {
        if (g_interrupted.load()) {
            std::fprintf(stderr, "\nasr: interrupted\n");
            break;
        }

        asr::result r;
        if (!asr::transcribe_file(*ctx, file, args.transcribe, args.output.no_prints,
                                  r, vad.get(), vp)) {
            ret = 1;
            continue;
        }

        // Suppress repetitions in the merged text.
        r.text = asr::suppress_repeats(r.text);
        for (auto & seg : r.segments) {
            seg.text = asr::suppress_repeats(seg.text);
        }

        const std::string base = args.output.out_base.empty() ? file : args.output.out_base;

        std::vector<asr::subtitle_cue> cues;
        if (args.output.out_srt || args.output.out_vtt) {
            cues = asr::split_cues(r);
        }

        if (args.output.out_txt) {
            std::ofstream f(base + ".txt");
            if (f) { asr::write_txt(f, r); if (!args.output.no_prints) std::fprintf(stderr, "asr: saved %s.txt\n", base.c_str()); }
            else { std::fprintf(stderr, "error: cannot write %s.txt\n", base.c_str()); ret = 1; }
        }
        if (args.output.out_json) {
            std::ofstream f(base + ".json");
            if (f) { asr::write_json_full(f, r); if (!args.output.no_prints) std::fprintf(stderr, "asr: saved %s.json\n", base.c_str()); }
            else { std::fprintf(stderr, "error: cannot write %s.json\n", base.c_str()); ret = 1; }
        }
        if (args.output.out_srt) {
            std::ofstream f(base + ".srt");
            if (f) { asr::write_srt(f, cues); if (!args.output.no_prints) std::fprintf(stderr, "asr: saved %s.srt (%zu cues)\n", base.c_str(), cues.size()); }
            else { std::fprintf(stderr, "error: cannot write %s.srt\n", base.c_str()); ret = 1; }
        }
        if (args.output.out_vtt) {
            std::ofstream f(base + ".vtt");
            if (f) { asr::write_vtt(f, cues); if (!args.output.no_prints) std::fprintf(stderr, "asr: saved %s.vtt (%zu cues)\n", base.c_str(), cues.size()); }
            else { std::fprintf(stderr, "error: cannot write %s.vtt\n", base.c_str()); ret = 1; }
        }
    }
    return ret;
#else
    std::fprintf(stderr,
                 "asr-cli built without engine (ASR_BUILD_ENGINE=OFF); "
                 "parsed %zu input file(s) OK.\n",
                 args.input_files.size());
    return 0;
#endif
}
