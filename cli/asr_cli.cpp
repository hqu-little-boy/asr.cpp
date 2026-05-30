#include "asr_args.h"

#include <cstdio>

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
    // Phase 2 wires the mtmd engine + driver here.
    std::fprintf(stderr, "asr-cli: engine path not yet implemented\n");
    return 0;
#else
    std::fprintf(stderr,
                 "asr-cli built without engine (ASR_BUILD_ENGINE=OFF); "
                 "parsed %zu input file(s) OK.\n",
                 args.input_files.size());
    return 0;
#endif
}
