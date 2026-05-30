#pragma once

#include <string>
#include <vector>

struct mtmd_context;

namespace asr {

// Load an audio file (wav/mp3/flac) via mtmd's helper, which decodes and
// resamples to mono float PCM at the model's required sample rate
// (mtmd_get_audio_sample_rate). The PCM is read back out of the resulting
// bitmap into `out_pcm`. Returns false on failure.
bool load_audio_pcm(mtmd_context * mctx, const std::string & path, std::vector<float> & out_pcm);

} // namespace asr
