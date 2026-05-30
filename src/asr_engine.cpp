#include "asr_engine.h"
#include "asr_profile.h"
#include "asr_gguf.h"
#include "asr_audio.h"

#include "common.h"
#include "sampling.h"
#include "chat.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace asr {

struct asr_context::impl {
    common_init_result_ptr    llama_init;
    llama_model *             model = nullptr;
    llama_context *           lctx  = nullptr;
    const llama_vocab *       vocab = nullptr;
    common_sampler *          smpl  = nullptr;
    llama_batch               batch{};
    int                       n_batch = 2048;

    mtmd::context_ptr         mctx;
    common_chat_templates_ptr tmpls;
    bool                      use_jinja = false;

    int                       sample_rate = 16000;
    const profile *           prof = nullptr;
    std::string               profile_name;

    ~impl() {
        if (smpl) {
            common_sampler_free(smpl);
        }
        llama_batch_free(batch);
    }
};

asr_context::asr_context() : p_(new impl()) {}
asr_context::~asr_context() = default;

std::unique_ptr<asr_context> asr_context::load(const model_params & mp, bool quiet) {
    static bool backends_loaded = false;
    if (!backends_loaded) {
        common_init();
        ggml_backend_load_all();
        backends_loaded = true;
    }
    // Mute logging before the model is loaded so --no-prints is actually quiet.
    if (quiet) {
        set_log_quiet();
    }

    common_params params;
    params.model.path     = mp.model;
    params.mmproj.path    = mp.mmproj;
    params.mmproj_use_gpu = mp.mmproj_use_gpu;
    params.n_gpu_layers   = mp.use_gpu ? -1 : 0; // -1 = auto offload, 0 = CPU
    if (mp.n_threads > 0) {
        params.cpuparams.n_threads = mp.n_threads;
    }

    std::unique_ptr<asr_context> self(new asr_context());
    impl & s = *self->p_;

    s.llama_init = common_init_from_params(params);
    if (!s.llama_init) {
        std::fprintf(stderr, "asr: failed to init from params\n");
        return nullptr;
    }
    s.model = s.llama_init->model();
    s.lctx  = s.llama_init->context();
    if (!s.model || !s.lctx) {
        std::fprintf(stderr, "asr: failed to load model '%s'\n", mp.model.c_str());
        return nullptr;
    }
    s.vocab   = llama_model_get_vocab(s.model);
    s.smpl    = common_sampler_init(s.model, params.sampling);
    s.batch   = llama_batch_init(1, 0, 1);
    s.n_batch = params.n_batch;

    if (!llama_model_chat_template(s.model, nullptr) && params.chat_template.empty()) {
        std::fprintf(stderr, "asr: model has no chat template\n");
        return nullptr;
    }
    s.tmpls     = common_chat_templates_init(s.model, params.chat_template);
    s.use_jinja = params.use_jinja;

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu         = mp.mmproj_use_gpu;
    mparams.print_timings   = false;
    mparams.n_threads       = mp.n_threads > 0 ? mp.n_threads : params.cpuparams.n_threads;
    mparams.flash_attn_type = params.flash_attn_type;
    mparams.warmup          = params.warmup;
    s.mctx.reset(mtmd_init_from_file(mp.mmproj.c_str(), s.model, mparams));
    if (!s.mctx.get()) {
        std::fprintf(stderr, "asr: failed to load mmproj '%s'\n", mp.mmproj.c_str());
        return nullptr;
    }
    if (!mtmd_support_audio(s.mctx.get())) {
        std::fprintf(stderr, "asr: mmproj '%s' has no audio support\n", mp.mmproj.c_str());
        return nullptr;
    }
    s.sample_rate = mtmd_get_audio_sample_rate(s.mctx.get());
    if (s.sample_rate <= 0) {
        s.sample_rate = 16000;
    }

    const std::string projector = read_projector_type(mp.mmproj);
    const std::string want      = mp.profile_override.empty() ? projector : mp.profile_override;
    s.prof         = &select_profile(want);
    s.profile_name = s.prof->name;
    if (!quiet) {
        std::fprintf(stderr, "asr: projector_type='%s', profile='%s', sample_rate=%d\n",
                     projector.c_str(), s.profile_name.c_str(), s.sample_rate);
    }

    return self;
}

int                 asr_context::sample_rate() const { return p_->sample_rate; }
const std::string & asr_context::profile_name() const { return p_->profile_name; }

bool asr_context::load_audio(const std::string & path, std::vector<float> & out_pcm) const {
    return load_audio_pcm(p_->mctx.get(), path, out_pcm);
}

chunk_text asr_context::transcribe_chunk(const std::vector<float> & pcm, const transcribe_params & tp) {
    impl &     s = *p_;
    chunk_text empty;
    if (pcm.empty()) {
        return empty;
    }

    // Independent context per chunk: clear KV cache and sampler state.
    llama_memory_clear(llama_get_memory(s.lctx), true);
    common_sampler_reset(s.smpl);

    // Build the user message (media marker + context) and apply the chat template.
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = s.prof->build_prompt(tp, mtmd_default_marker());
    std::vector<common_chat_msg> history;
    const std::string formatted =
        common_chat_format_single(s.tmpls.get(), history, msg, /*add_ass*/ true, s.use_jinja);

    mtmd_input_text text;
    text.text          = formatted.c_str();
    text.add_special   = true;
    text.parse_special = true;

    mtmd::bitmap bmp(mtmd_bitmap_init_from_audio(pcm.size(), pcm.data()));
    if (!bmp.ptr) {
        return empty;
    }
    mtmd::bitmaps bitmaps;
    bitmaps.entries.push_back(std::move(bmp));
    auto c_ptr = bitmaps.c_ptr();

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (mtmd_tokenize(s.mctx.get(), chunks.ptr.get(), &text, c_ptr.data(), c_ptr.size()) != 0) {
        std::fprintf(stderr, "asr: mtmd_tokenize failed\n");
        return empty;
    }

    llama_pos new_n_past = 0;
    if (mtmd_helper_eval_chunks(s.mctx.get(), s.lctx, chunks.ptr.get(),
                                /*n_past*/ 0, /*seq_id*/ 0, s.n_batch,
                                /*logits_last*/ true, &new_n_past)) {
        std::fprintf(stderr, "asr: failed to eval audio chunks\n");
        return empty;
    }
    llama_pos n_past = new_n_past;

    int n_predict = tp.n_predict;
    if (n_predict <= 0) {
        const double seconds = (double) pcm.size() / std::max(1, s.sample_rate);
        n_predict = std::max(256, (int) (seconds * 64.0));
    }

    std::string raw;
    for (int i = 0; i < n_predict; ++i) {
        const llama_token tok = common_sampler_sample(s.smpl, s.lctx, -1);
        common_sampler_accept(s.smpl, tok, true);
        if (llama_vocab_is_eog(s.vocab, tok)) {
            break;
        }
        raw += common_token_to_piece(s.lctx, tok, /*special*/ true);

        common_batch_clear(s.batch);
        common_batch_add(s.batch, tok, n_past++, {0}, true);
        if (llama_decode(s.lctx, s.batch)) {
            std::fprintf(stderr, "asr: llama_decode failed\n");
            break;
        }
    }

    return s.prof->parse_output(raw);
}

namespace {
void quiet_log_cb(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}
} // namespace

void set_log_quiet() {
    ggml_log_set(quiet_log_cb, nullptr);
    llama_log_set(quiet_log_cb, nullptr);
    mtmd_helper_log_set(quiet_log_cb, nullptr); // also routes mtmd_log_set
}

} // namespace asr
