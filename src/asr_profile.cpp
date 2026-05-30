#include "asr_profile.h"

#include <cctype>
#include <string>

namespace asr {

namespace {

std::string trim(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) ++b;
    while (e > b && std::isspace((unsigned char) s[e - 1])) --e;
    return s.substr(b, e - b);
}

// Case-insensitive check whether `s` begins with `prefix`.
bool istarts_with(const std::string & s, const char * prefix) {
    for (size_t i = 0; prefix[i] != '\0'; ++i) {
        if (i >= s.size()) return false;
        if (std::tolower((unsigned char) s[i]) != std::tolower((unsigned char) prefix[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

chunk_text parse_qwen3a_output(const std::string & raw) {
    static const std::string kMarker  = "<asr_text>";
    static const std::string kKeyword = "language";

    chunk_text out;

    const size_t pos = raw.find(kMarker);
    if (pos == std::string::npos) {
        // No ASR marker: treat the whole input as transcription text.
        out.text = trim(raw);
        return out;
    }

    // Preamble before the first marker: mine for a leading "language <value>".
    const std::string preamble = trim(raw.substr(0, pos));
    if (istarts_with(preamble, kKeyword.c_str())) {
        const std::string rest = preamble.substr(kKeyword.size());
        // Require a whitespace separator between the keyword and the value.
        if (!rest.empty() && std::isspace((unsigned char) rest[0])) {
            out.language = trim(rest);
        }
    }

    // Transcription is everything after the first marker.
    out.text = trim(raw.substr(pos + kMarker.size()));
    return out;
}

namespace {

// All current profiles build the same user message (marker + context); they
// differ only in how they parse the model's raw output.
std::string build_marker_plus_context(const transcribe_params & params,
                                       const std::string &       media_marker) {
    return media_marker + params.context;
}

const profile & generic_profile() {
    static const profile p{
        "generic",
        build_marker_plus_context,
        [](const std::string & raw) { return chunk_text{trim(raw), std::string()}; },
    };
    return p;
}

const profile & qwen3a_profile() {
    static const profile p{
        "qwen3a",
        build_marker_plus_context,
        [](const std::string & raw) { return parse_qwen3a_output(raw); },
    };
    return p;
}

} // namespace

const profile & select_profile(const std::string & projector_type) {
    if (projector_type == "qwen3a") {
        return qwen3a_profile();
    }
    return generic_profile();
}

} // namespace asr
