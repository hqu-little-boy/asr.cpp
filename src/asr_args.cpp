#include "asr_args.h"
#include "asr.h"

#include <string>

namespace asr {

namespace {

// Parse a full integer string. Returns false if not entirely numeric.
bool to_int(const std::string & s, int & out) {
    try {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

// Parse a full float string. Returns false if not entirely numeric.
bool to_float(const std::string & s, float & out) {
    try {
        size_t pos = 0;
        float v = std::stof(s, &pos);
        if (pos != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace

cli_args parse_args(int argc, const char * const * argv) {
    cli_args a;

    // Fetch the value following a value-taking flag, or flag an error.
    auto need = [&](int & i, const std::string & flag) -> const char * {
        if (i + 1 >= argc) {
            a.error = true;
            a.error_msg = "missing value for " + flag;
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc && !a.error; ++i) {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            a.help = true;
        } else if (arg == "-m" || arg == "--model") {
            if (const char * v = need(i, arg)) a.model.model = v;
        } else if (arg == "--mmproj") {
            if (const char * v = need(i, arg)) a.model.mmproj = v;
        } else if (arg == "-f" || arg == "--file") {
            if (const char * v = need(i, arg)) a.input_files.emplace_back(v);
        } else if (arg == "-of" || arg == "--output-file") {
            if (const char * v = need(i, arg)) a.output.out_base = v;
        } else if (arg == "-otxt") {
            a.output.out_txt = true;
        } else if (arg == "-oj") {
            a.output.out_json = true;
        } else if (arg == "-np" || arg == "--no-prints") {
            a.output.no_prints = true;
        } else if (arg == "-ng" || arg == "--no-gpu") {
            a.model.use_gpu = false;
            a.model.mmproj_use_gpu = false;
        } else if (arg == "--context") {
            if (const char * v = need(i, arg)) a.transcribe.context = v;
        } else if (arg == "--profile") {
            if (const char * v = need(i, arg)) a.model.profile_override = v;
        } else if (arg == "--carry-context") {
            a.transcribe.carry_context = true;
        } else if (arg == "-t" || arg == "--threads") {
            if (const char * v = need(i, arg)) {
                if (!to_int(v, a.model.n_threads)) {
                    a.error = true;
                    a.error_msg = "invalid integer for " + arg + ": " + v;
                }
            }
        } else if (arg == "-n" || arg == "--n-predict") {
            if (const char * v = need(i, arg)) {
                if (!to_int(v, a.transcribe.n_predict)) {
                    a.error = true;
                    a.error_msg = "invalid integer for " + arg + ": " + v;
                }
            }
        } else if (arg == "--chunk-length") {
            if (const char * v = need(i, arg)) {
                if (!to_float(v, a.transcribe.chunk_length_s)) {
                    a.error = true;
                    a.error_msg = "invalid number for " + arg + ": " + v;
                }
            }
        } else if (arg == "-") {
            // stdin convention: keep as a literal input token.
            a.input_files.emplace_back(arg);
        } else if (!arg.empty() && arg[0] == '-') {
            a.error = true;
            a.error_msg = "unknown argument: " + arg;
        } else {
            a.input_files.emplace_back(arg);
        }
    }

    // Validate required options (skipped on --help or an earlier error).
    if (!a.help && !a.error) {
        if (a.model.model.empty()) {
            a.error = true;
            a.error_msg = "missing required --model";
        } else if (a.model.mmproj.empty()) {
            a.error = true;
            a.error_msg = "missing required --mmproj";
        } else if (a.input_files.empty()) {
            a.error = true;
            a.error_msg = "no input audio files given";
        }
    }

    return a;
}

std::string usage_string(const char * argv0) {
    std::string s;
    s += "asr-cli (asr.cpp ";
    s += version();
    s += ") - speech recognition via llama.cpp/mtmd\n\n";
    s += "usage: ";
    s += argv0;
    s += " [options] file0 [file1 ...]\n\n";
    s += "  -m,   --model FNAME       main GGUF model (required)\n";
    s += "        --mmproj FNAME      multimodal projector GGUF (required)\n";
    s += "  -f,   --file FNAME        input audio file (wav/mp3/flac)\n";
    s += "  -of,  --output-file BASE  output file base path (no extension)\n";
    s += "  -otxt                     write .txt transcription\n";
    s += "  -oj                       write .json transcription\n";
    s += "  -np,  --no-prints         only print results\n";
    s += "  -t,   --threads N         number of threads\n";
    s += "  -ng,  --no-gpu            disable GPU\n";
    s += "        --context TEXT      context / hotwords bias\n";
    s += "        --chunk-length N    chunk length in seconds\n";
    s += "        --profile NAME      override model profile\n";
    s += "        --carry-context     feed prior transcript as context (experimental)\n";
    s += "  -n,   --n-predict N       max tokens generated per chunk\n";
    s += "  -h,   --help              show this help message\n";
    return s;
}

} // namespace asr
