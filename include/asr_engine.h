#pragma once

#include "asr.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace asr {

// The mtmd-backed ASR engine: owns the llama model, mtmd context, sampler and
// chat templates, and the selected per-model profile. One instance is reused
// across chunks; each transcribe_chunk() resets the context so chunks are
// independent.
class asr_context {
  public:
    // Load model + mmproj and select the profile. Returns nullptr on failure.
    // When quiet is true, ggml/llama/mtmd logging is muted to error-only.
    // Sampling params from tp (temperature, top_p, repeat_penalty) are applied
    // to the sampler; values < 0 use the model defaults.
    static std::unique_ptr<asr_context> load(const model_params & mp,
                                             bool quiet = false,
                                             const transcribe_params & tp = transcribe_params{});
    ~asr_context();

    asr_context(const asr_context &)             = delete;
    asr_context & operator=(const asr_context &) = delete;

    // Transcribe one chunk of mono PCM (at sample_rate()) in an independent
    // context. Returns {text, language} parsed by the profile.
    chunk_text transcribe_chunk(std::span<const float> pcm, const transcribe_params & tp);
    chunk_text transcribe_chunk(const std::vector<float> & pcm, const transcribe_params & tp);

    // Create a new independent inference context sharing the same model.
    // Each clone gets its own llama_context, sampler, and batch — suitable
    // for parallel processing from a separate thread.
    std::unique_ptr<asr_context> clone() const;

    // Decode an audio file to mono PCM at the model's sample rate.
    bool load_audio(const std::string & path, std::vector<float> & out_pcm) const;

    int                 sample_rate() const;
    const std::string & profile_name() const;

  private:
    asr_context();
    struct impl;
    std::unique_ptr<impl> p_;
};

// Route ggml / llama / mtmd logging to error-only (used for --no-prints to mute
// the verbose per-chunk encode/decode messages).
void set_log_quiet();
void set_log_default();

} // namespace asr
