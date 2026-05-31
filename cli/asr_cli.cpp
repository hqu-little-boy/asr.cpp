#include "asr_args.h"

#include <cstdio>
#include <csignal>

#ifdef ASR_WITH_ENGINE
#include "asr_engine.h"
#include "asr_driver.h"
#include "asr_job.h"
#include "asr_vad.h"

#include <atomic>
#include <string>
#include <utility>

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
                                  r, vad.get(), vp, args.processors)) {
            ret = 1;
            continue;
        }

        auto finalized = asr::finalize_transcription(std::move(r), args.output, file);
        if (!finalized) {
            const auto & err = finalized.error();
            std::fprintf(stderr, "error: %s: %s\n",
                         err.path.string().c_str(), err.message.c_str());
            ret = 1;
            continue;
        }
        if (!args.output.no_prints) {
            for (const auto & item : finalized->outputs) {
                if (item.cue_count > 0) {
                    std::fprintf(stderr, "asr: saved %s (%zu cues)\n",
                                 item.path.string().c_str(), item.cue_count);
                } else {
                    std::fprintf(stderr, "asr: saved %s\n", item.path.string().c_str());
                }
            }
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
