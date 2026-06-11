// asr_vad.cpp — FireRedVAD (DFSMN) runtime, ported from CrispASR (MIT).
//
// The DFSMN is tiny (~588K params); it runs entirely on CPU with plain loops —
// no ggml compute graph. ggml/gguf is used ONLY to read the F32 weights out of
// the GGUF at load time (loaded into a CPU ggml_context, then copied into
// std::vector<float> and the context is freed).
//
// Feature extraction and the forward pass are copied byte-for-byte from the
// reference so probabilities match: int16 fbank scaling, Hann^0.85 window,
// pre-emphasis 0.97, per-frame DC removal, CMVN, and double accumulation.

#include "asr_vad.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace asr {

namespace {

struct hparams {
    int R = 8, H = 256, P = 128, N1 = 20, N2 = 20, S1 = 1, S2 = 1, idim = 80, odim = 1;
};

struct fsmn_block {
    std::vector<float> fc1_w, fc1_b; // [H, P] + [H]
    std::vector<float> fc2_w;        // [P, H] (no bias)
    std::vector<float> lb_w, la_w;   // lookback/ahead [P, N]
};

struct firered_model {
    hparams hp;
    std::vector<float> fc1_w, fc1_b;       // [H, idim] + [H]
    std::vector<float> fc2_w, fc2_b;       // [P, H] + [P]
    std::vector<float> fsmn1_lb, fsmn1_la; // first FSMN lookback/ahead
    std::vector<fsmn_block> blocks;        // R-1 blocks
    std::vector<float> dnn_w, dnn_b;       // [H, P] + [H]
    std::vector<float> out_w, out_b;       // [1, H] + [1]
    std::vector<float> cmvn_mean, cmvn_std;
};

uint32_t kv_u32(const gguf_context * g, const char * key, uint32_t def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(g, id);
        default:               return def;
    }
}

// Load all F32 weights from the GGUF directly (no ggml backend needed).
bool load_model(const std::string & path, firered_model & m) {
    ggml_context *   meta = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = false; // allocate + read tensor data into `meta`
    gp.ctx      = &meta;

    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        return false;
    }

    hparams & hp = m.hp;
    hp.R    = (int) kv_u32(g, "firered_vad.R", hp.R);
    hp.H    = (int) kv_u32(g, "firered_vad.H", hp.H);
    hp.P    = (int) kv_u32(g, "firered_vad.P", hp.P);
    hp.N1   = (int) kv_u32(g, "firered_vad.N1", hp.N1);
    hp.N2   = (int) kv_u32(g, "firered_vad.N2", hp.N2);
    hp.S1   = (int) kv_u32(g, "firered_vad.S1", hp.S1);
    hp.S2   = (int) kv_u32(g, "firered_vad.S2", hp.S2);
    hp.idim = (int) kv_u32(g, "firered_vad.idim", hp.idim);
    hp.odim = (int) kv_u32(g, "firered_vad.odim", hp.odim);

    bool ok = true;
    auto rd = [&](const char * name, std::vector<float> & v) {
        ggml_tensor * t = ggml_get_tensor(meta, name);
        if (t == nullptr || t->type != GGML_TYPE_F32) {
            ok = false;
            return;
        }
        const int n = (int) ggml_nelements(t);
        v.resize(n);
        std::memcpy(v.data(), t->data, (size_t) n * sizeof(float));
    };

    rd("dfsmn.fc1.0.weight", m.fc1_w);
    rd("dfsmn.fc1.0.bias",   m.fc1_b);
    rd("dfsmn.fc2.0.weight", m.fc2_w);
    rd("dfsmn.fc2.0.bias",   m.fc2_b);
    rd("dfsmn.fsmn1.lookback_filter.weight",  m.fsmn1_lb);
    rd("dfsmn.fsmn1.lookahead_filter.weight", m.fsmn1_la);

    const int n_blocks = hp.R - 1; // first FSMN is separate
    m.blocks.resize(std::max(0, n_blocks));
    for (int i = 0; i < n_blocks; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fc1.0.weight", i);                rd(buf, m.blocks[i].fc1_w);
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fc1.0.bias", i);                  rd(buf, m.blocks[i].fc1_b);
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fc2.weight", i);                  rd(buf, m.blocks[i].fc2_w);
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fsmn.lookback_filter.weight", i); rd(buf, m.blocks[i].lb_w);
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fsmn.lookahead_filter.weight", i);rd(buf, m.blocks[i].la_w);
    }

    rd("dfsmn.dnns.0.weight", m.dnn_w);
    rd("dfsmn.dnns.0.bias",   m.dnn_b);
    rd("out.weight",          m.out_w);
    rd("out.bias",            m.out_b);
    rd("cmvn.mean",           m.cmvn_mean);
    rd("cmvn.std",            m.cmvn_std);

    gguf_free(g);
    ggml_free(meta);
    return ok;
}

// out[t,n] = sum_k x[t,k]*w[n,k] + b[n]   (double accumulation, as in reference)
void cpu_linear(const float * x, const float * w, const float * b, float * out, int T, int K, int N) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int t = 0; t < T; t++) {
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) {
                s += x[t * K + k] * w[n * K + k];
            }
            out[t * N + n] = (float) s + (b ? b[n] : 0.0f);
        }
    }
}

void cpu_relu(float * x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

// FSMN: memory = x + lookback_conv(x) + lookahead_conv(x), replicating PyTorch
// depthwise Conv1d padding/trim semantics exactly.
void cpu_fsmn(const float * x, float * out, const float * lb_w, const float * la_w,
              int T, int P, int N1, int S1, int N2, int S2) {
    std::vector<float> x_pt((size_t) P * T);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int p = 0; p < P; p++)
        for (int t = 0; t < T; t++)
            x_pt[p * T + t] = x[t * P + p];

    const int lb_pad     = (N1 - 1) * S1;
    const int lb_out_len = T + lb_pad;
    std::vector<float> lb_conv((size_t) P * lb_out_len, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int p = 0; p < P; p++) {
        for (int t_out = 0; t_out < lb_out_len; t_out++) {
            float s = 0;
            for (int k = 0; k < N1; k++) {
                int t_in = t_out - lb_pad + k * S1;
                if (t_in >= 0 && t_in < T) s += lb_w[p * N1 + k] * x_pt[p * T + t_in];
            }
            lb_conv[p * lb_out_len + t_out] = s;
        }
    }

    memcpy(out, x, (size_t) T * P * sizeof(float)); // residual
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int p = 0; p < P; p++)
        for (int t = 0; t < T; t++)
            out[t * P + p] += lb_conv[p * lb_out_len + t];

    if (N2 > 0 && T > 1) {
        const int la_pad     = (N2 - 1) * S2;
        const int la_out_len = T + la_pad;
        std::vector<float> la_conv((size_t) P * la_out_len, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int p = 0; p < P; p++) {
            for (int t_out = 0; t_out < la_out_len; t_out++) {
                float s = 0;
                for (int k = 0; k < N2; k++) {
                    int t_in = t_out - la_pad + k * S2;
                    if (t_in >= 0 && t_in < T) s += la_w[p * N2 + k] * x_pt[p * T + t_in];
                }
                la_conv[p * la_out_len + t_out] = s;
            }
        }
        const int skip = N2 * S2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int p = 0; p < P; p++)
            for (int t = 0; t < T; t++) {
                int la_idx = skip + t;
                if (la_idx < la_out_len) out[t * P + p] += la_conv[p * la_out_len + la_idx];
            }
    }
}

// 80-dim log-mel fbank (matches firered_asr). int16-scaled input.
void compute_fbank(const float * pcm, int n_samples, std::vector<float> & features, int & n_frames) {
    const int   n_fft = 512, hop = 160, win = 400, n_mels = 80, sr = 16000;
    const float preemph = 0.97f, low_freq = 20.0f, high_freq = (float) sr / 2;

    n_frames = (n_samples - win) / hop + 1;
    if (n_frames <= 0) {
        n_frames = 0;
        return;
    }

    const int bins = n_fft / 2 + 1;
    std::vector<float> mel_fb((size_t) n_mels * bins, 0.0f);
    {
        auto hz2mel = [](float hz) { return 1127.0f * logf(1.0f + hz / 700.0f); };
        auto mel2hz = [](float mm) { return 700.0f * (expf(mm / 1127.0f) - 1.0f); };
        float ml = hz2mel(low_freq), mh = hz2mel(high_freq);
        std::vector<float> c(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++) c[i] = mel2hz(ml + i * (mh - ml) / (n_mels + 1));
        for (int m = 0; m < n_mels; m++)
            for (int k = 0; k < bins; k++) {
                float f = (float) k * sr / n_fft;
                if (f > c[m] && f <= c[m + 1] && c[m + 1] > c[m])
                    mel_fb[m * bins + k] = (f - c[m]) / (c[m + 1] - c[m]);
                else if (f > c[m + 1] && f < c[m + 2] && c[m + 2] > c[m + 1])
                    mel_fb[m * bins + k] = (c[m + 2] - f) / (c[m + 2] - c[m + 1]);
            }
    }
    std::vector<float> window(win);
    for (int i = 0; i < win; i++) {
        float hh = 0.5f - 0.5f * cosf(2.0f * (float) M_PI * i / (win - 1));
        window[i] = powf(hh, 0.85f);
    }
    features.resize((size_t) n_frames * n_mels);
    std::vector<float> fre(n_fft), fim(n_fft);
    auto fft = [](float * r, float * im, int n) {
        for (int i = 1, j = 0; i < n; i++) {
            int b = n >> 1;
            for (; j & b; b >>= 1) j ^= b;
            j ^= b;
            if (i < j) { std::swap(r[i], r[j]); std::swap(im[i], im[j]); }
        }
        for (int l = 2; l <= n; l <<= 1) {
            float a = -2.f * (float) M_PI / l, wr = cosf(a), wi = sinf(a);
            for (int i = 0; i < n; i += l) {
                float cr = 1, ci = 0;
                for (int j = 0; j < l / 2; j++) {
                    float t = r[i + j + l / 2] * cr - im[i + j + l / 2] * ci,
                          u = r[i + j + l / 2] * ci + im[i + j + l / 2] * cr;
                    r[i + j + l / 2] = r[i + j] - t;
                    im[i + j + l / 2] = im[i + j] - u;
                    r[i + j] += t;
                    im[i + j] += u;
                    float nr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = nr;
                }
            }
        }
    };
    const float scale_to_i16 = 32768.0f; // model trained on int16 fbank

    for (int t = 0; t < n_frames; t++) {
        int off = t * hop;
        std::vector<float> fr(win);
        float dc = 0;
        for (int i = 0; i < win; i++) {
            fr[i] = ((off + i < n_samples) ? pcm[off + i] : 0.0f) * scale_to_i16;
            dc += fr[i];
        }
        dc /= win;
        for (int i = 0; i < win; i++) fr[i] -= dc;
        for (int i = win - 1; i > 0; i--) fr[i] -= preemph * fr[i - 1];
        fr[0] -= preemph * fr[0];
        std::fill(fre.begin(), fre.end(), 0.0f);
        std::fill(fim.begin(), fim.end(), 0.0f);
        for (int i = 0; i < win; i++) fre[i] = fr[i] * window[i];
        fft(fre.data(), fim.data(), n_fft);
        for (int m = 0; m < n_mels; m++) {
            float s = 0;
            for (int k = 0; k < bins; k++) s += (fre[k] * fre[k] + fim[k] * fim[k]) * mel_fb[m * bins + k];
            features[t * n_mels + m] = logf(std::max(s, 1.1920929e-7f));
        }
    }
}

// Run fbank + CMVN + DFSMN forward, returning per-frame speech probabilities.
std::vector<float> forward_probs(const firered_model & m, const float * pcm, int n_samples, std::atomic<bool> * cancel) {
    const hparams & hp = m.hp;

    std::vector<float> features;
    int n_frames = 0;
    compute_fbank(pcm, n_samples, features, n_frames);
    if (n_frames <= 0) return {};

    if (!m.cmvn_mean.empty()) {
        for (int t = 0; t < n_frames; t++)
            for (int f = 0; f < hp.idim; f++)
                features[t * hp.idim + f] = (features[t * hp.idim + f] - m.cmvn_mean[f]) / m.cmvn_std[f];
    }

    const int T = n_frames;

    std::vector<float> h((size_t) T * hp.H);
    cpu_linear(features.data(), m.fc1_w.data(), m.fc1_b.data(), h.data(), T, hp.idim, hp.H);
    cpu_relu(h.data(), T * hp.H);

    std::vector<float> p((size_t) T * hp.P);
    cpu_linear(h.data(), m.fc2_w.data(), m.fc2_b.data(), p.data(), T, hp.H, hp.P);
    cpu_relu(p.data(), T * hp.P);

    std::vector<float> mem((size_t) T * hp.P);
    cpu_fsmn(p.data(), mem.data(), m.fsmn1_lb.data(), m.fsmn1_la.data(), T, hp.P, hp.N1, hp.S1, hp.N2, hp.S2);

    std::vector<float> tmp_h((size_t) T * hp.H), tmp_p((size_t) T * hp.P), tmp_mem((size_t) T * hp.P);
    for (const auto & b : m.blocks) {
        if (cancel && cancel->load()) return {};
        cpu_linear(mem.data(), b.fc1_w.data(), b.fc1_b.data(), tmp_h.data(), T, hp.P, hp.H);
        cpu_relu(tmp_h.data(), T * hp.H);
        cpu_linear(tmp_h.data(), b.fc2_w.data(), nullptr, tmp_p.data(), T, hp.H, hp.P);
        cpu_fsmn(tmp_p.data(), tmp_mem.data(), b.lb_w.data(), b.la_w.data(), T, hp.P, hp.N1, hp.S1, hp.N2, hp.S2);
        for (int j = 0; j < T * hp.P; j++) mem[j] = tmp_mem[j] + mem[j]; // skip connection
    }

    cpu_linear(mem.data(), m.dnn_w.data(), m.dnn_b.data(), h.data(), T, hp.P, hp.H);
    cpu_relu(h.data(), T * hp.H);

    std::vector<float> probs(T);
    cpu_linear(h.data(), m.out_w.data(), m.out_b.data(), probs.data(), T, hp.H, 1);
    for (int t = 0; t < T; t++) probs[t] = 1.0f / (1.0f + expf(-probs[t]));
    return probs;
}

} // namespace

struct vad_context::impl {
    firered_model model;
};

vad_context::vad_context() : p_(new impl()) {}
vad_context::~vad_context() = default;

std::unique_ptr<vad_context> vad_context::load(const std::string & model_path) {
    std::unique_ptr<vad_context> self(new vad_context());
    if (!load_model(model_path, self->p_->model)) {
        return nullptr;
    }
    return self;
}

std::vector<float> vad_context::frame_probs(const float * pcm, int n_samples, std::atomic<bool> * cancel) const {
    if (!pcm || n_samples <= 0) return {};
    return forward_probs(p_->model, pcm, n_samples, cancel);
}

std::vector<vad_segment> vad_context::detect(const float * pcm, int n_samples, const vad_params & params, std::atomic<bool> * cancel) const {
    std::vector<vad_segment> segs;
    if (!pcm || n_samples <= 0) return segs;

    const std::vector<float> probs = forward_probs(p_->model, pcm, n_samples, cancel);
    if (cancel && cancel->load()) return segs;
    const int T = (int) probs.size();
    if (T <= 0) return segs;

    const float frame_sec         = 0.01f; // 10 ms per frame
    const int   min_speech_frames = (int) (params.min_speech_sec / frame_sec);
    const int   min_silence_frames = (int) (params.min_silence_sec / frame_sec);

    bool  in_speech     = false;
    float seg_start     = 0;
    int   speech_count  = 0;
    int   silence_count = 0;
    for (int t = 0; t < T; t++) {
        if (probs[t] >= params.threshold) {
            speech_count++;
            silence_count = 0;
            if (!in_speech && speech_count >= min_speech_frames) {
                in_speech = true;
                seg_start = (t - speech_count + 1) * frame_sec;
            }
        } else {
            silence_count++;
            speech_count = 0;
            if (in_speech && silence_count >= min_silence_frames) {
                in_speech = false;
                vad_segment seg;
                seg.start_sec = seg_start;
                seg.end_sec   = (t - silence_count + 1) * frame_sec;
                if (seg.end_sec - seg.start_sec >= params.min_speech_sec) segs.push_back(seg);
            }
        }
    }
    if (in_speech) {
        vad_segment seg;
        seg.start_sec = seg_start;
        seg.end_sec   = T * frame_sec;
        segs.push_back(seg);
    }
    return segs;
}

} // namespace asr
