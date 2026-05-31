#include "asr_job.h"

#include "asr_postprocess.h"

namespace asr {

std::expected<finalized_transcription, asr_error>
finalize_transcription(result r,
                       const output_params & output,
                       const std::filesystem::path & input_path) {
    r.text = suppress_repeats(r.text);
    for (auto & seg : r.segments) {
        seg.text = suppress_repeats(seg.text);
    }

    auto written = write_selected_outputs(r, output, input_path);
    if (!written) {
        return std::unexpected(written.error());
    }

    finalized_transcription out;
    out.transcript = std::move(r);
    out.outputs    = std::move(*written);
    return out;
}

} // namespace asr
