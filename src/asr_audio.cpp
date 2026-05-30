#include "asr_audio.h"

#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstring>

namespace asr {

bool load_audio_pcm(mtmd_context * mctx, const std::string & path, std::vector<float> & out_pcm) {
    out_pcm.clear();

    mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(mctx, path.c_str());
    if (bmp == nullptr) {
        return false;
    }

    bool ok = false;
    if (mtmd_bitmap_is_audio(bmp)) {
        const size_t          n_bytes = mtmd_bitmap_get_n_bytes(bmp);
        const unsigned char * data    = mtmd_bitmap_get_data(bmp);
        const size_t          n       = n_bytes / sizeof(float);
        if (n > 0 && data != nullptr) {
            out_pcm.resize(n);
            std::memcpy(out_pcm.data(), data, n * sizeof(float));
            ok = true;
        }
    }

    mtmd_bitmap_free(bmp);
    return ok;
}

} // namespace asr
