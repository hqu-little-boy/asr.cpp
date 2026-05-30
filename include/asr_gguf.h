#pragma once

#include <string>

namespace asr {

// Read the projector type (e.g. "qwen3a") from an mmproj GGUF's
// "clip.projector_type" metadata key, using only the public gguf API.
// Returns "" if the file cannot be read or the key is absent / not a string.
std::string read_projector_type(const std::string & mmproj_path);

} // namespace asr
