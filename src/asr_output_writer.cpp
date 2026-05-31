#include "asr_output_writer.h"

#include "asr_output.h"

#include <fstream>
#include <ostream>

namespace asr {

std::string_view output_extension(output_format format) {
    switch (format) {
        case output_format::txt:  return "txt";
        case output_format::json: return "json";
        case output_format::srt:  return "srt";
        case output_format::vtt:  return "vtt";
        case output_format::lrc:  return "lrc";
        case output_format::csv:  return "csv";
    }
    return "";
}

std::string_view output_format_name(output_format format) {
    return output_extension(format);
}

std::vector<output_format> selected_output_formats(const output_params & params) {
    std::vector<output_format> formats;
    if (params.out_txt)  formats.push_back(output_format::txt);
    if (params.out_json) formats.push_back(output_format::json);
    if (params.out_srt)  formats.push_back(output_format::srt);
    if (params.out_vtt)  formats.push_back(output_format::vtt);
    if (params.out_lrc)  formats.push_back(output_format::lrc);
    if (params.out_csv)  formats.push_back(output_format::csv);
    return formats;
}

namespace {

bool needs_cues(output_format format) {
    return format == output_format::srt ||
           format == output_format::vtt ||
           format == output_format::lrc ||
           format == output_format::csv;
}

std::filesystem::path output_base_path(const output_params & params,
                                       const std::filesystem::path & input_path) {
    if (!params.out_base.empty()) {
        return std::filesystem::path(params.out_base);
    }
    return input_path;
}

std::filesystem::path output_path_for(const std::filesystem::path & base,
                                      output_format format) {
    std::filesystem::path path = base;
    path += ".";
    path += output_extension(format);
    return path;
}

void write_one(std::ostream & os,
               output_format format,
               const result & r,
               const std::vector<subtitle_cue> & cues) {
    switch (format) {
        case output_format::txt:  write_txt(os, r); break;
        case output_format::json: write_json_full(os, r); break;
        case output_format::srt:  write_srt(os, cues); break;
        case output_format::vtt:  write_vtt(os, cues); break;
        case output_format::lrc:  write_lrc(os, cues); break;
        case output_format::csv:  write_csv(os, cues); break;
    }
}

} // namespace

std::expected<std::vector<written_output>, asr_error>
write_selected_outputs(const result & r,
                       const output_params & params,
                       const std::filesystem::path & input_path) {
    const std::vector<output_format> formats = selected_output_formats(params);
    std::vector<written_output> written;
    written.reserve(formats.size());

    bool any_cue_format = false;
    for (const output_format format : formats) {
        any_cue_format = any_cue_format || needs_cues(format);
    }
    const std::vector<subtitle_cue> cues = any_cue_format ? split_cues(r) : std::vector<subtitle_cue>{};
    const std::filesystem::path base = output_base_path(params, input_path);

    for (const output_format format : formats) {
        const std::filesystem::path path = output_path_for(base, format);
        std::ofstream file(path);
        if (!file) {
            return std::unexpected(make_error(error_code::output,
                                              "cannot write output file",
                                              "write_outputs",
                                              path));
        }

        write_one(file, format, r, cues);
        if (!file) {
            return std::unexpected(make_error(error_code::output,
                                              "failed while writing output file",
                                              "write_outputs",
                                              path));
        }

        written_output item;
        item.format    = format;
        item.path      = path;
        item.cue_count = needs_cues(format) ? cues.size() : 0;
        written.push_back(std::move(item));
    }

    return written;
}

} // namespace asr
