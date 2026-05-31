#include "asr_args.h"
#include "asr.h"

#include <CLI/CLI.hpp>

#include <fstream>
#include <string>
#include <vector>

namespace asr {

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

void normalize_legacy_options(std::vector<std::string> & args) {
    for (std::string & arg : args) {
        if      (arg == "-of")   arg = "--output-file";
        else if (arg == "-otxt") arg = "--out-txt";
        else if (arg == "-oj")   arg = "--out-json";
        else if (arg == "-osrt") arg = "--out-srt";
        else if (arg == "-ovtt") arg = "--out-vtt";
        else if (arg == "-olrc") arg = "--out-lrc";
        else if (arg == "-ocsv") arg = "--out-csv";
        else if (arg == "-np")   arg = "--no-prints";
        else if (arg == "-ng")   arg = "--no-gpu";
    }
}

void add_output_format(cli_args & a, const std::string & fmt) {
    if (fmt == "txt" || fmt == "text") { a.output.out_txt = true; }
    else if (fmt == "json")            { a.output.out_json = true; }
    else if (fmt == "srt")             { a.output.out_srt = true; }
    else if (fmt == "vtt")             { a.output.out_vtt = true; }
    else if (fmt == "lrc")             { a.output.out_lrc = true; }
    else if (fmt == "csv")             { a.output.out_csv = true; }
    else {
        a.error = true;
        a.error_msg = "unknown format: " + fmt;
    }
}

std::string cli11_error_message(const CLI::ParseError & e) {
    const std::string msg = e.what();
    if (e.get_name() == "ArgumentMismatch") {
        const std::string missing = " missing";
        const std::string required = " required ";
        const size_t colon = msg.find(':');
        if (colon != std::string::npos &&
            msg.find(required, colon) != std::string::npos &&
            msg.rfind(missing) == msg.size() - missing.size()) {
            return "missing value for " + msg.substr(0, colon);
        }
    }
    const std::string one = "The following argument was not expected: ";
    const std::string many = "The following arguments were not expected: ";
    const size_t one_pos = msg.find(one);
    const size_t many_pos = msg.find(many);
    if (one_pos != std::string::npos) return "unknown argument: " + msg.substr(one_pos + one.size());
    if (many_pos != std::string::npos) return "unknown argument: " + msg.substr(many_pos + many.size());
    return msg;
}
} // namespace

cli_args parse_args(int argc, const char * const * argv) {
    cli_args a;

    // Expand @responsefiles first.
    std::vector<std::string> expanded = expand_response_files(argc, argv);
    normalize_legacy_options(expanded);
    const int exp_argc = (int) expanded.size();
    // Build a c-string array for the rest of the parser.
    std::vector<const char *> exp_argv(exp_argc);
    for (int i = 0; i < exp_argc; ++i) exp_argv[i] = expanded[i].c_str();

    CLI::App app{"asr.cpp speech recognition"};
    app.set_help_flag("-h,--help", "show this help message");

    app.add_option("-m,--model", a.model.model, "main GGUF model");
    app.add_option("--mmproj", a.model.mmproj, "multimodal projector GGUF");
    app.add_option_function<std::string>("-f,--file", [&](const std::string & path) {
        a.input_files.emplace_back(path);
    }, "input audio file")->trigger_on_parse();
    app.add_option("--output-file", a.output.out_base, "output file base path");
    app.add_flag("--out-txt", a.output.out_txt, "write .txt transcription");
    app.add_flag("--out-json", a.output.out_json, "write .json transcription");
    app.add_flag("--out-srt", a.output.out_srt, "write .srt subtitles");
    app.add_flag("--out-vtt", a.output.out_vtt, "write .vtt subtitles");
    app.add_flag("--out-lrc", a.output.out_lrc, "write .lrc lyrics");
    app.add_flag("--out-csv", a.output.out_csv, "write .csv");
    app.add_option_function<std::string>("--output-format", [&](const std::string & fmt) {
        add_output_format(a, fmt);
    }, "output format")->trigger_on_parse();
    app.add_flag("--vad", a.vad.use_vad, "use FireRedVAD");
    app.add_option("--vad-model", a.vad.model_path, "FireRedVAD GGUF model");
    app.add_option("--vad-threshold", a.vad.threshold, "VAD speech probability threshold");
    app.add_option("--vad-min-speech", a.vad.min_speech_sec, "minimum speech duration");
    app.add_option("--vad-min-silence", a.vad.min_silence_sec, "minimum silence duration");
    app.add_flag("--no-prints", a.output.no_prints, "only print results");
    app.add_flag_callback("--no-gpu", [&]() {
        a.model.use_gpu = false;
        a.model.mmproj_use_gpu = false;
    }, "disable GPU");
    app.add_option("--context", a.transcribe.context, "context / hotwords bias");
    app.add_option("-l,--language", a.transcribe.language, "force language");
    app.add_option("--profile", a.model.profile_override, "override model profile");
    app.add_flag("--carry-context", a.transcribe.carry_context, "feed prior transcript as context");
    app.add_option("--temperature", a.transcribe.temperature, "sampling temperature");
    app.add_option("--top-p", a.transcribe.top_p, "top-p sampling");
    app.add_option("--repeat-penalty", a.transcribe.repeat_penalty, "repeat penalty");
    app.add_option("-t,--threads", a.model.n_threads, "number of threads per inference");
    app.add_option("-p,--processors", a.processors, "number of parallel inference instances");
    app.add_option("-n,--n-predict", a.transcribe.n_predict, "max tokens generated per chunk");
    app.add_option("--chunk-length", a.transcribe.chunk_length_s, "chunk length in seconds");
    app.add_option_function<std::string>("input", [&](const std::string & path) {
        a.input_files.emplace_back(path);
    }, "input audio file")->expected(-1)->trigger_on_parse();

    try {
        app.parse(exp_argc, exp_argv.data());
    } catch (const CLI::CallForHelp &) {
        a.help = true;
    } catch (const CLI::ParseError & e) {
        a.error = true;
        a.error_msg = cli11_error_message(e);
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
        } else if (!a.vad.use_vad &&
                   (a.output.out_json || a.output.out_srt || a.output.out_vtt ||
                    a.output.out_lrc  || a.output.out_csv)) {
            a.error = true;
            a.error_msg = "output formats other than txt require --vad";
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
