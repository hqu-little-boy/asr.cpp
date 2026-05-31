#include "asr_profile.h"

#include <cctype>
#include <string>
#include <string_view>

namespace asr {

namespace {

std::string_view trim_view(std::string_view s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) ++b;
    while (e > b && std::isspace((unsigned char) s[e - 1])) --e;
    return s.substr(b, e - b);
}

// Case-insensitive check whether `s` begins with `prefix`.
bool istarts_with(std::string_view s, std::string_view prefix) {
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (i >= s.size()) return false;
        if (std::tolower((unsigned char) s[i]) != std::tolower((unsigned char) prefix[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

chunk_text parse_qwen3a_output(const std::string & raw) {
    static constexpr std::string_view kMarker  = "<asr_text>";
    static constexpr std::string_view kKeyword = "language";

    chunk_text out;
    const std::string_view input(raw);

    const size_t pos = input.find(kMarker);
    if (pos == std::string_view::npos) {
        // No ASR marker: treat the whole input as transcription text.
        out.text = std::string(trim_view(input));
        return out;
    }

    // Preamble before the first marker: mine for a leading "language <value>".
    const std::string_view preamble = trim_view(input.substr(0, pos));
    if (istarts_with(preamble, kKeyword)) {
        const std::string_view rest = preamble.substr(kKeyword.size());
        // Require a whitespace separator between the keyword and the value.
        if (!rest.empty() && std::isspace((unsigned char) rest[0])) {
            out.language = std::string(trim_view(rest));
        }
    }

    // Transcription is everything after the first marker.
    out.text = std::string(trim_view(input.substr(pos + kMarker.size())));
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
        [](const std::string & raw) { return chunk_text{std::string(trim_view(raw)), std::string()}; },
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
