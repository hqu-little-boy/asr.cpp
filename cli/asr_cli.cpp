#include "asr.h"

#include <cstdio>
#include <string>

namespace {

void print_usage(const char * argv0) {
    std::printf(
        "asr-cli (asr.cpp %s) - speech recognition via llama.cpp/mtmd\n\n"
        "usage: %s [options] file0 [file1 ...]\n\n"
        "  -m,   --model FNAME       main GGUF model (required)\n"
        "        --mmproj FNAME      multimodal projector GGUF (required)\n"
        "  -f,   --file FNAME        input audio file (wav/mp3/flac)\n"
        "  -of,  --output-file BASE  output file base path (no extension)\n"
        "  -otxt                     write .txt transcription\n"
        "  -oj                       write .json transcription\n"
        "  -np,  --no-prints         only print results\n"
        "  -t,   --threads N         number of threads\n"
        "  -ng,  --no-gpu            disable GPU\n"
        "        --context TEXT      context / hotwords bias\n"
        "        --chunk-length N    chunk length in seconds\n"
        "        --profile NAME      override model profile\n"
        "  -n,   --n-predict N       max tokens generated per chunk\n"
        "  -h,   --help              show this help message\n",
        asr::version(), argv0);
}

} // namespace

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Full argument parsing (Phase 1d) and transcription (Phase 2) land later.
    print_usage(argv[0]);
    return 0;
}
