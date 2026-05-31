#pragma once

#include "asr.h"
#include "asr_error.h"
#include "asr_output_writer.h"

#include <expected>
#include <filesystem>
#include <vector>

namespace asr {

struct finalized_transcription {
    result                      transcript;
    std::vector<written_output> outputs;
};

std::expected<finalized_transcription, asr_error>
finalize_transcription(result r,
                       const output_params & output,
                       const std::filesystem::path & input_path);

} // namespace asr
