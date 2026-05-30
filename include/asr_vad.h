#pragma once

#include <memory>
#include <string>
#include <vector>

namespace asr {

// A detected speech segment, in seconds.
struct vad_segment {
    float start_sec = 0.0f;
    float end_sec   = 0.0f;
};

// VAD thresholding / hysteresis parameters.
struct vad_params {
    float threshold        = 0.5f;  // per-frame speech probability threshold
    float min_speech_sec   = 0.25f; // minimum speech run to emit a segment
    float min_silence_sec  = 0.10f; // silence run needed to end a segment
};

// FireRedVAD (DFSMN) voice-activity detector. Ported from CrispASR (MIT);
// model weights are FireRedTeam/FireRedVAD (Apache-2.0). Runs entirely on CPU
// with plain loops; ggml/gguf is used only to read F32 weights at load time.
class vad_context {
  public:
    // Load a firered-vad GGUF. Returns nullptr on failure.
    static std::unique_ptr<vad_context> load(const std::string & model_path);
    ~vad_context();

    vad_context(const vad_context &)             = delete;
    vad_context & operator=(const vad_context &) = delete;

    // Detect speech segments in 16 kHz mono PCM.
    std::vector<vad_segment> detect(const float * pcm, int n_samples, const vad_params & p) const;

    // Per-frame (10 ms) speech probabilities — exposed for testing/debugging.
    std::vector<float> frame_probs(const float * pcm, int n_samples) const;

  private:
    vad_context();
    struct impl;
    std::unique_ptr<impl> p_;
};

} // namespace asr
