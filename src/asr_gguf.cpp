#include "asr_gguf.h"

#include "gguf.h"

namespace asr {

namespace {
std::string get_str_key(const gguf_context * ctx, const char * key) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id >= 0 && gguf_get_kv_type(ctx, key_id) == GGUF_TYPE_STRING) {
        const char * val = gguf_get_val_str(ctx, key_id);
        if (val != nullptr) {
            return val;
        }
    }
    return std::string();
}
} // namespace

std::string read_projector_type(const std::string & mmproj_path) {
    gguf_init_params params{};
    params.no_alloc = true;   // metadata only; do not load tensor data
    params.ctx      = nullptr;

    gguf_context * ctx = gguf_init_from_file(mmproj_path.c_str(), params);
    if (ctx == nullptr) {
        return std::string();
    }

    // Audio mmprojs namespace the type under "clip.audio.projector_type";
    // older / combined files use the bare "clip.projector_type".
    std::string result;
    for (const char * key : {"clip.audio.projector_type", "clip.projector_type"}) {
        result = get_str_key(ctx, key);
        if (!result.empty()) {
            break;
        }
    }

    gguf_free(ctx);
    return result;
}

} // namespace asr
