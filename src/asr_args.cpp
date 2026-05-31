#include "asr_args.h"
#include "asr.h"

#include <fstream>
#include <string>
#include <vector>

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

namespace {
// Expand @file tokens: each line in the file becomes a separate argument.
std::vector<std::string> expand_response_files(int argc, const char * const * argv) {
    std::vector<std::string> expanded;
    expanded.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.size() > 1 && arg[0] == '@') {
            const std::string path = arg.substr(1);
            std::ifstream f(path);
            if (!f) {
                // Keep the @token as-is; parse_args will report an error later
                // if it looks like an unknown flag.
                expanded.push_back(arg);
                continue;
            }
            std::string line;
            while (std::getline(f, line)) {
                // Strip trailing \r (Windows line endings).
                if (!line.empty() && line.back() == '\r') line.pop_back();
                // Skip empty lines and comments (#).
                if (line.empty() || line[0] == '#') continue;
                expanded.push_back(line);
            }
        } else {
            expanded.push_back(arg);
        }
    }
    return expanded;
}
} // namespace

cli_args parse_args(int argc, const char * const * argv) {
    cli_args a;

    // Expand @responsefiles first.
    const std::vector<std::string> expanded = expand_response_files(argc, argv);
    const int exp_argc = (int) expanded.size();
    // Build a c-string array for the rest of the parser.
    std::vector<const char *> exp_argv(exp_argc);
    for (int i = 0; i < exp_argc; ++i) exp_argv[i] = expanded[i].c_str();

    // Fetch the value following a value-taking flag, or flag an error.
    auto need = [&](int & i, const std::string & flag) -> const char * {
        if (i + 1 >= exp_argc) {
            a.error = true;
            a.error_msg = "missing value for " + flag;
            return nullptr;
        }
        return exp_argv[++i];
    };

    for (int i = 1; i < exp_argc && !a.error; ++i) {
        const std::string arg = exp_argv[i];

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
        } else if (arg == "-osrt") {
            a.output.out_srt = true;
        } else if (arg == "-ovtt") {
            a.output.out_vtt = true;
        } else if (arg == "-olrc") {
            a.output.out_lrc = true;
        } else if (arg == "-ocsv") {
            a.output.out_csv = true;
        } else if (arg == "--output-format") {
            if (const char * v = need(i, arg)) {
                const std::string fmt = v;
                if (fmt == "txt" || fmt == "text")    { a.output.out_txt  = true; }
                else if (fmt == "json")                { a.output.out_json = true; }
                else if (fmt == "srt")                 { a.output.out_srt  = true; }
                else if (fmt == "vtt")                 { a.output.out_vtt  = true; }
                else if (fmt == "lrc")                 { a.output.out_lrc  = true; }
                else if (fmt == "csv")                 { a.output.out_csv  = true; }
                else { a.error = true; a.error_msg = "unknown format: " + fmt; }
            }
        } else if (arg == "--vad") {
            a.vad.use_vad = true;
        } else if (arg == "--vad-model") {
            if (const char * v = need(i, arg)) a.vad.model_path = v;
        } else if (arg == "--vad-threshold") {
            if (const char * v = need(i, arg)) {
                if (!to_float(v, a.vad.threshold)) { a.error = true; a.error_msg = "invalid number for " + arg + ": " + v; }
            }
        } else if (arg == "--vad-min-speech") {
            if (const char * v = need(i, arg)) {
                if (!to_float(v, a.vad.min_speech_sec)) { a.error = true; a.error_msg = "invalid number for " + arg + ": " + v; }
            }
        } else if (arg == "--vad-min-silence") {
            if (const char * v = need(i, arg)) {
                if (!to_float(v, a.vad.min_silence_sec)) { a.error = true; a.error_msg = "invalid number for " + arg + ": " + v; }
            }
        } else if (arg == "-np" || arg == "--no-prints") {
            a.output.no_prints = true;
        } else if (arg == "-ng" || arg == "--no-gpu") {
            a.model.use_gpu = false;
            a.model.mmproj_use_gpu = false;
        } else if (arg == "--context") {
            if (const char * v = need(i, arg)) a.transcribe.context = v;
        } else if (arg == "-l" || arg == "--language") {
            if (const char * v = need(i, arg)) a.transcribe.language = v;
        } else if (arg == "--profile") {
            if (const char * v = need(i, arg)) a.model.profile_override = v;
        } else if (arg == "--carry-context") {
            a.transcribe.carry_context = true;
        } else if (arg == "--temperature") {
            if (const char * v = need(i, arg)) {
                if (!to_float(v, a.transcribe.temperature)) { a.error = true; a.error_msg = "invalid number for " + arg + ": " + v; }
            }
        } else if (arg == "--top-p") {
            if (const char * v = need(i, arg)) {
                if (!to_float(v, a.transcribe.top_p)) { a.error = true; a.error_msg = "invalid number for " + arg + ": " + v; }
            }
        } else if (arg == "--repeat-penalty") {
            if (const char * v = need(i, arg)) {
                if (!to_float(v, a.transcribe.repeat_penalty)) { a.error = true; a.error_msg = "invalid number for " + arg + ": " + v; }
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (const char * v = need(i, arg)) {
                if (!to_int(v, a.model.n_threads)) {
                    a.error = true;
                    a.error_msg = "invalid integer for " + arg + ": " + v;
                }
            }
        } else if (arg == "-p" || arg == "--processors") {
            if (const char * v = need(i, arg)) {
                if (!to_int(v, a.processors)) {
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
        } else if (a.vad.use_vad && a.vad.model_path.empty()) {
            a.error = true;
            a.error_msg = "--vad requires --vad-model";
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
    s += "  -osrt                     write .srt subtitles\n";
    s += "  -ovtt                     write .vtt subtitles\n";
    s += "  -olrc                     write .lrc lyrics\n";
    s += "  -ocsv                     write .csv (start,end,text)\n";
    s += "        --output-format FMT  output format: txt|json|srt|vtt|lrc|csv\n";
    s += "  -np,  --no-prints         only print results\n";
    s += "  -t,   --threads N         number of threads per inference\n";
    s += "  -p,   --processors N      number of parallel inference instances\n";
    s += "  -ng,  --no-gpu            disable GPU\n";
    s += "        --context TEXT      context / hotwords bias\n";
    s += "  -l,   --language LANG    force language (skip auto-detection)\n";
    s += "        --chunk-length N    chunk length in seconds\n";
    s += "        --profile NAME      override model profile\n";
    s += "        --carry-context     feed prior transcript as context (experimental)\n";
    s += "  -n,   --n-predict N       max tokens generated per chunk\n";
    s += "        --temperature N     sampling temperature (default: model)\n";
    s += "        --top-p N           top-p sampling (default: model)\n";
    s += "        --repeat-penalty N  repeat penalty (default: model)\n";
    s += "        --vad               use FireRedVAD for speech segmentation\n";
    s += "        --vad-model FNAME   FireRedVAD GGUF model (required with --vad)\n";
    s += "        --vad-threshold N   VAD speech probability threshold (default 0.5)\n";
    s += "        --vad-min-speech N  minimum speech duration in seconds (default 0.25)\n";
    s += "        --vad-min-silence N minimum silence duration in seconds (default 0.10)\n";
    s += "  -h,   --help              show this help message\n";
    s += "        @file               read arguments from file (one per line, # comments)\n";
    return s;
}

} // namespace asr
