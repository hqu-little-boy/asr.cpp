#include "asr_engine.h"
#include "asr_profile.h"
#include "asr_gguf.h"
#include "asr_audio.h"

#include "common.h"
#include "log.h"
#include "sampling.h"
#include "chat.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstdio>
#include <span>
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

    // Stored for clone(): allow creating new worker contexts sharing the model.
    model_params              stored_mp;
    common_params             stored_params;
    bool                      quiet = false;
    std::string               stored_chat_template; // for recreating in clone()

    ~impl() {
        if (smpl) {
            common_sampler_free(smpl);
        }
        llama_batch_free(batch);
    }
};

asr_context::asr_context() : p_(new impl()) {}
asr_context::~asr_context() = default;

std::unique_ptr<asr_context> asr_context::load(const model_params & mp, bool quiet,
                                               const transcribe_params & tp) {
    static bool backends_loaded = false;
    if (!backends_loaded) {
        common_init();
        ggml_backend_load_all();
        backends_loaded = true;
    }
    // Configure logging before the model is loaded.
    if (quiet) {
        set_log_quiet();
    } else {
        set_log_default();
    }

    common_params params;
    params.model.path     = mp.model;
    params.mmproj.path    = mp.mmproj;
    params.mmproj_use_gpu = mp.mmproj_use_gpu;
    params.n_gpu_layers   = mp.use_gpu ? -1 : 0; // -1 = auto offload, 0 = CPU
    if (mp.n_threads > 0) {
        params.cpuparams.n_threads = mp.n_threads;
    }

    // Apply user-specified sampling params (< 0 = use default).
    if (tp.temperature >= 0)    params.sampling.temp           = tp.temperature;
    if (tp.top_p >= 0)          params.sampling.top_p          = tp.top_p;
    if (tp.repeat_penalty >= 0) params.sampling.penalty_repeat = tp.repeat_penalty;

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

    // Store params for clone().
    s.stored_mp    = mp;
    s.stored_params = params;
    s.quiet        = quiet;
    s.stored_chat_template = params.chat_template;

    return self;
}

int                 asr_context::sample_rate() const { return p_->sample_rate; }
const std::string & asr_context::profile_name() const { return p_->profile_name; }

std::unique_ptr<asr_context> asr_context::clone() const {
    impl & s = *p_;
    std::unique_ptr<asr_context> worker(new asr_context());
    impl & w = *worker->p_;

    // Share read-only resources.
    w.model        = s.model;
    w.vocab        = s.vocab;
    w.use_jinja    = s.use_jinja;
    w.sample_rate  = s.sample_rate;
    w.prof         = s.prof;
    w.profile_name = s.profile_name;
    w.n_batch      = s.n_batch;

    // Create new chat templates (unique_ptr, can't share).
    w.tmpls = common_chat_templates_init(s.model, s.stored_chat_template);

    // Create a new mtmd context (not thread-safe — each worker needs its own).
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu         = s.stored_mp.mmproj_use_gpu;
    mparams.print_timings   = false;
    mparams.n_threads       = s.stored_params.cpuparams.n_threads;
    mparams.flash_attn_type = s.stored_params.flash_attn_type;
    mparams.warmup          = false; // skip warmup for clones
    w.mctx.reset(mtmd_init_from_file(s.stored_mp.mmproj.c_str(), s.model, mparams));
    if (!w.mctx.get()) {
        std::fprintf(stderr, "asr: clone failed to create mtmd context\n");
        return nullptr;
    }

    // Create a new llama context from the shared model.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_batch  = s.n_batch;
    cparams.n_ctx    = llama_n_ctx(s.lctx);
    cparams.flash_attn_type = s.stored_params.flash_attn_type;
    w.lctx = llama_init_from_model(s.model, cparams);
    if (!w.lctx) {
        std::fprintf(stderr, "asr: clone failed to create llama context\n");
        return nullptr;
    }

    w.smpl  = common_sampler_init(s.model, s.stored_params.sampling);
    w.batch = llama_batch_init(1, 0, 1);

    return worker;
}

bool asr_context::load_audio(const std::string & path, std::vector<float> & out_pcm) const {
    return load_audio_pcm(p_->mctx.get(), path, out_pcm);
}

chunk_text asr_context::transcribe_chunk(std::span<const float> pcm, const transcribe_params & tp) {
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
    std::string formatted =
        common_chat_format_single(s.tmpls.get(), history, msg, /*add_ass*/ true, s.use_jinja);

    // When --language is set, prefill "language <X><asr_text>" to skip the
    // model's auto-detection preamble and get pure transcription output.
    const bool language_forced = !tp.language.empty();
    if (language_forced) {
        formatted += "language " + tp.language + "<asr_text>";
    }

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

    // When language was forced via prefill, the model outputs pure transcription
    // (no "language X<asr_text>" prefix). Otherwise, parse the protocol.
    if (language_forced) {
        chunk_text ct;
        // Trim leading/trailing whitespace.
        size_t b = 0, e = raw.size();
        while (b < e && std::isspace((unsigned char) raw[b])) ++b;
        while (e > b && std::isspace((unsigned char) raw[e - 1])) --e;
        ct.text     = raw.substr(b, e - b);
        ct.language = tp.language;
        return ct;
    }
    return s.prof->parse_output(raw);
}

chunk_text asr_context::transcribe_chunk(const std::vector<float> & pcm, const transcribe_params & tp) {
    return transcribe_chunk(std::span<const float>(pcm), tp);
}

namespace {
void quiet_log_cb(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

void default_log_cb(enum ggml_log_level level, const char * text, void * /*user_data*/) {
#ifdef NDEBUG
    if (level == GGML_LOG_LEVEL_DEBUG) {
        return;
    }
#endif
    std::fputs(text, stderr);
}
} // namespace

void set_log_quiet() {
    ggml_log_set(quiet_log_cb, nullptr);
    llama_log_set(quiet_log_cb, nullptr);
    mtmd_helper_log_set(quiet_log_cb, nullptr); // also routes mtmd_log_set
    common_log_set_verbosity_thold(1); // LOG_LEVEL_ERROR only (suppress INFO/WARN)
}

void set_log_default() {
    ggml_log_set(default_log_cb, nullptr);
    llama_log_set(default_log_cb, nullptr);
    mtmd_helper_log_set(default_log_cb, nullptr);
}

} // namespace asr
