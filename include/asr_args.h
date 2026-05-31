#pragma once

#include "asr.h"
#include "asr_error.h"

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace asr {

// Fully parsed command line. parse_args never exits and never throws; callers
// inspect `help` / `error` and act accordingly.
struct cli_args {
    model_params             model;
    transcribe_params        transcribe;
    output_params            output;
    vad_run_params           vad;
    std::vector<std::string> input_files;

    int         processors = 1; // -p/--processors: parallel inference instances
    bool        help  = false; // -h/--help was given
    bool        error = false; // a parse or validation error occurred
    std::string error_msg;     // human-readable error (when error == true)
};

// Runtime-facing configuration derived from CLI syntax. It intentionally uses
// filesystem paths and drops parser-only fields such as `help` and `error`.
struct run_config {
    model_params                    model;
    transcribe_params               transcribe;
    output_params                   output;
    vad_run_params                  vad;
    std::vector<std::filesystem::path> input_files;
    int                             processors = 1;
};

// Parse argv (argv[0] is the program name). On a value/flag error or a missing
// required option (--model, --mmproj, at least one input) sets error+error_msg.
// When --help is present, `help` is set and required-option validation is
// skipped.
cli_args parse_args(int argc, const char * const * argv);

// C++23 parse API: returns a value or a structured error instead of encoding
// parse failures inside cli_args.
std::expected<cli_args, asr_error> parse_args_checked(int argc, const char * const * argv);

// Convert parsed CLI arguments into runtime configuration. Call only after
// successful parsing.
run_config to_run_config(const cli_args & args);

// The --help / usage text.
std::string usage_string(const char * argv0);

} // namespace asr
