// test_vad.cpp — FireRedVAD (asr_vad) tests.
//
// Model-gated tests require ASR_RUN_MODEL_TESTS=1 and
// models/firered-vad-GGUF/firered-vad.gguf in the source tree.

#include "asr_vad.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef ASR_SOURCE_DIR
#define ASR_SOURCE_DIR "."
#endif

namespace {
constexpr int kSR = 16000;

std::string vad_model_path() {
    return std::string(ASR_SOURCE_DIR) + "/models/firered-vad-GGUF/firered-vad.gguf";
}

bool model_exists() {
    return std::getenv("ASR_RUN_MODEL_TESTS") &&
           !asr::vad_context::load(vad_model_path()).operator bool() == false &&
           asr::vad_context::load(vad_model_path()) != nullptr;
}

// Append n samples of silence (zeros) to out.
void append_silence(std::vector<float> & out, size_t n) {
    out.insert(out.end(), n, 0.0f);
}

// Append n samples of a sine tone (amplitude 0.5, 440 Hz) to out.
void append_tone(std::vector<float> & out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out.push_back(0.5f * std::sin(2.0 * M_PI * 440.0 * (double) i / kSR));
    }
}

} // namespace

// ---- Always-run tests (no model required) ----

TEST(Vad, MissingFileReturnsNull) {
    auto ctx = asr::vad_context::load("/no/such/file.gguf");
    EXPECT_EQ(ctx, nullptr);
}

TEST(Vad, DetectOnNullContext) {
    // Can't call detect on a nullptr, but we can verify load fails gracefully.
    auto ctx = asr::vad_context::load("/no/such/file.gguf");
    EXPECT_EQ(ctx, nullptr);
}

// ---- Model-gated tests (require ASR_RUN_MODEL_TESTS=1) ----

TEST(Vad, LoadModel) {
    if (!std::getenv("ASR_RUN_MODEL_TESTS")) {
        GTEST_SKIP() << "set ASR_RUN_MODEL_TESTS=1 to run model-dependent tests";
    }
    auto ctx = asr::vad_context::load(vad_model_path());
    ASSERT_NE(ctx, nullptr) << "failed to load " << vad_model_path();
}

TEST(Vad, SilenceNoSegments) {
    if (!std::getenv("ASR_RUN_MODEL_TESTS")) {
        GTEST_SKIP() << "set ASR_RUN_MODEL_TESTS=1 to run model-dependent tests";
    }
    auto ctx = asr::vad_context::load(vad_model_path());
    ASSERT_NE(ctx, nullptr);

    // 3 seconds of pure silence → no speech segments.
    std::vector<float> pcm;
    append_silence(pcm, kSR * 3);
    asr::vad_params p;
    auto segs = ctx->detect(pcm.data(), (int) pcm.size(), p);
    EXPECT_TRUE(segs.empty());
}

TEST(Vad, SilenceNoSegmentsLong) {
    if (!std::getenv("ASR_RUN_MODEL_TESTS")) {
        GTEST_SKIP() << "set ASR_RUN_MODEL_TESTS=1 to run model-dependent tests";
    }
    auto ctx = asr::vad_context::load(vad_model_path());
    ASSERT_NE(ctx, nullptr);

    // 5 seconds of pure silence → still no speech segments (regression guard).
    std::vector<float> pcm;
    append_silence(pcm, kSR * 5);
    asr::vad_params p;
    p.min_speech_sec = 0.1f; // very permissive; silence should still yield nothing
    auto segs = ctx->detect(pcm.data(), (int) pcm.size(), p);
    EXPECT_TRUE(segs.empty());
}

TEST(Vad, FrameProbsRange) {
    if (!std::getenv("ASR_RUN_MODEL_TESTS")) {
        GTEST_SKIP() << "set ASR_RUN_MODEL_TESTS=1 to run model-dependent tests";
    }
    auto ctx = asr::vad_context::load(vad_model_path());
    ASSERT_NE(ctx, nullptr);

    // Mixed signal: 1s tone + 1s silence + 1s tone.
    std::vector<float> pcm;
    append_tone(pcm, kSR);
    append_silence(pcm, kSR);
    append_tone(pcm, kSR);
    auto probs = ctx->frame_probs(pcm.data(), (int) pcm.size());
    EXPECT_FALSE(probs.empty());
    for (float v : probs) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
    }
}

TEST(Vad, SegmentBoundsContainment) {
    if (!std::getenv("ASR_RUN_MODEL_TESTS")) {
        GTEST_SKIP() << "set ASR_RUN_MODEL_TESTS=1 to run model-dependent tests";
    }
    auto ctx = asr::vad_context::load(vad_model_path());
    ASSERT_NE(ctx, nullptr);

    std::vector<float> pcm;
    append_tone(pcm, kSR * 2);
    asr::vad_params p;
    p.min_speech_sec  = 0.1f;
    p.min_silence_sec = 0.05f;
    auto segs = ctx->detect(pcm.data(), (int) pcm.size(), p);
    for (const auto & s : segs) {
        EXPECT_GE(s.start_sec, 0.0f);
        EXPECT_LE(s.end_sec, (float) pcm.size() / kSR);
        EXPECT_GE(s.end_sec - s.start_sec, p.min_speech_sec - 0.01f); // allow rounding
    }
}

// The real test: VAD on a real speech file (P1, ~9 min). Requires the model
// AND the test-data audio file. Uses ASR_TEST_AUDIO if set, otherwise falls
// back to the P1 test-data wav.
TEST(Vad, RealAudioDetectsSpeech) {
    if (!std::getenv("ASR_RUN_MODEL_TESTS")) {
        GTEST_SKIP() << "set ASR_RUN_MODEL_TESTS=1 to run model-dependent tests";
    }
    auto ctx = asr::vad_context::load(vad_model_path());
    ASSERT_NE(ctx, nullptr);

    const char * audio_env = std::getenv("ASR_TEST_AUDIO");
    std::string audio_path = audio_env
        ? std::string(audio_env)
        : std::string(ASR_SOURCE_DIR) + "/test-data/[P1]她赢得了世界，却输掉了自己【桂冠之下】.wav";

    // Load audio via mtmd helper (reuse the engine's load_audio for convenience).
    // For a standalone VAD test we use mtmd directly.
    // If the audio file doesn't exist, skip gracefully.
    FILE * f = fopen(audio_path.c_str(), "rb");
    if (!f) {
        GTEST_SKIP() << "audio file not found: " << audio_path;
    }
    fclose(f);

    // We can't use mtmd here (asr_vad links only ggml). Use a simple approach:
    // read the wav manually (16-bit PCM header + data).
    f = fopen(audio_path.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> raw(sz);
    fread(raw.data(), 1, sz, f);
    fclose(f);

    // Parse WAV: find the "data" chunk.
    const unsigned char * pcm_data = nullptr;
    size_t pcm_bytes = 0;
    for (size_t i = 0; i + 4 < raw.size(); i++) {
        if (raw[i] == 'd' && raw[i+1] == 'a' && raw[i+2] == 't' && raw[i+3] == 'a') {
            uint32_t chunk_sz = *(uint32_t*)(raw.data() + i + 4);
            pcm_data = raw.data() + i + 8;
            pcm_bytes = std::min((size_t) chunk_sz, raw.size() - i - 8);
            break;
        }
    }
    ASSERT_NE(pcm_data, nullptr) << "could not find 'data' chunk in WAV";

    // Convert 16-bit PCM to float [-1, 1], take first channel only.
    size_t n_samples = pcm_bytes / 4; // stereo 16-bit → mono float
    const int16_t * s16 = (const int16_t *) pcm_data;
    std::vector<float> pcm(n_samples);
    for (size_t i = 0; i < n_samples; i++) pcm[i] = s16[i * 2] / 32768.0f; // left channel

    // Truncate to ~30s to keep the test fast.
    if (pcm.size() > (size_t) kSR * 30) pcm.resize(kSR * 30);

    asr::vad_params p;
    auto segs = ctx->detect(pcm.data(), (int) pcm.size(), p);
    EXPECT_FALSE(segs.empty()) << "VAD should find speech in a real speech file";
    EXPECT_GE(segs.size(), 3u) << "expected multiple speech segments in 30s of speech";
    for (const auto & s : segs) {
        EXPECT_GE(s.start_sec, 0.0f);
        EXPECT_LE(s.end_sec, (float) pcm.size() / kSR + 0.01f);
    }
}
