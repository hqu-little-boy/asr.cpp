#pragma once

#include "asr.h"
#include "asr_error.h"

#include <expected>
#include <filesystem>
#include <string_view>
#include <vector>

namespace asr {

enum class output_format {
    txt,
    json,
    srt,
    vtt,
    lrc,
    csv,
};

struct written_output {
    output_format         format = output_format::txt;
    std::filesystem::path path;
    size_t                cue_count = 0;
};

std::string_view output_extension(output_format format);
std::string_view output_format_name(output_format format);

std::vector<output_format> selected_output_formats(const output_params & params);

std::expected<std::vector<written_output>, asr_error>
write_selected_outputs(const result & r,
                       const output_params & params,
                       const std::filesystem::path & input_path);

} // namespace asr
