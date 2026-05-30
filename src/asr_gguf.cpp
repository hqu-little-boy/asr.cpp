#include "asr_gguf.h"

#include "gguf.h"

namespace asr {

std::string read_projector_type(const std::string & mmproj_path) {
    gguf_init_params params{};
    params.no_alloc = true;   // metadata only; do not load tensor data
    params.ctx      = nullptr;

    gguf_context * ctx = gguf_init_from_file(mmproj_path.c_str(), params);
    if (ctx == nullptr) {
        return std::string();
    }

    std::string result;
    const int64_t key_id = gguf_find_key(ctx, "clip.projector_type");
    if (key_id >= 0 && gguf_get_kv_type(ctx, key_id) == GGUF_TYPE_STRING) {
        const char * val = gguf_get_val_str(ctx, key_id);
        if (val != nullptr) {
            result = val;
        }
    }

    gguf_free(ctx);
    return result;
}

} // namespace asr
