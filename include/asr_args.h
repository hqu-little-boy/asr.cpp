#pragma once

#include "asr.h"

#include <string>
#include <vector>

namespace asr {

// Fully parsed command line. parse_args never exits and never throws; callers
// inspect `help` / `error` and act accordingly.
struct cli_args {
    model_params             model;
    transcribe_params        transcribe;
    output_params            output;
    std::vector<std::string> input_files;

    bool        help  = false; // -h/--help was given
    bool        error = false; // a parse or validation error occurred
    std::string error_msg;     // human-readable error (when error == true)
};

// Parse argv (argv[0] is the program name). On a value/flag error or a missing
// required option (--model, --mmproj, at least one input) sets error+error_msg.
// When --help is present, `help` is set and required-option validation is
// skipped.
cli_args parse_args(int argc, const char * const * argv);

// The --help / usage text.
std::string usage_string(const char * argv0);

} // namespace asr
