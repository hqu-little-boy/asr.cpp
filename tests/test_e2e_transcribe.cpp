#include "asr_engine.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#ifndef ASR_SOURCE_DIR
#define ASR_SOURCE_DIR "."
#endif

namespace {
std::string env_or(const char * key, const std::string & fallback) {
    const char * v = std::getenv(key);
    return v ? std::string(v) : fallback;
}
} // namespace

// End-to-end transcription on a short clip of real audio. Gated behind
// ASR_RUN_MODEL_TESTS (needs the ~800MB model). Asserts language + non-empty
// rather than exact text, to stay robust to minor decoding nondeterminism.
TEST(E2E, TranscribeShortClip) {
    if (!std::getenv("ASR_RUN_MODEL_TESTS")) {
        GTEST_SKIP() << "set ASR_RUN_MODEL_TESTS=1 to run model-dependent tests";
    }

    asr::model_params mp;
    mp.model  = env_or("ASR_TEST_MODEL",  std::string(ASR_SOURCE_DIR) + "/models/Qwen3-ASR-0.6B-Q8_0.gguf");
    mp.mmproj = env_or("ASR_TEST_MMPROJ", std::string(ASR_SOURCE_DIR) + "/models/mmproj-Qwen3-ASR-0.6B-bf16.gguf");

    auto ctx = asr::asr_context::load(mp);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->profile_name(), "qwen3a");

    const std::string audio =
        env_or("ASR_TEST_AUDIO",
               std::string(ASR_SOURCE_DIR) + "/test-data/[P1]她赢得了世界，却输掉了自己【桂冠之下】.wav");

    std::vector<float> pcm;
    ASSERT_TRUE(ctx->load_audio(audio, pcm)) << "could not load audio: " << audio;

    const int    sr  = ctx->sample_rate();
    const size_t cap = (size_t) sr * 20; // truncate to ~20s to keep the test fast
    if (pcm.size() > cap) {
        pcm.resize(cap);
    }

    const asr::chunk_text ct = ctx->transcribe_chunk(pcm, asr::transcribe_params{});
    EXPECT_FALSE(ct.text.empty());
    EXPECT_EQ(ct.language, "Chinese");
}

// Verify Qwen3-ASR-1.7B works with the same qwen3a profile (different model,
// same projector type). Gated behind ASR_RUN_MODEL_TESTS_17B.
TEST(E2E, Transcribe17BModel) {
    if (!std::getenv("ASR_RUN_MODEL_TESTS_17B")) {
        GTEST_SKIP() << "set ASR_RUN_MODEL_TESTS_17B=1 to run 1.7B model tests";
    }

    asr::model_params mp;
    mp.model  = env_or("ASR_TEST_MODEL_17B",  std::string(ASR_SOURCE_DIR) + "/models/Qwen3-ASR-1.7B-Q8_0.gguf");
    mp.mmproj = env_or("ASR_TEST_MMPROJ_17B", std::string(ASR_SOURCE_DIR) + "/models/mmproj-Qwen3-ASR-1.7B-bf16.gguf");

    auto ctx = asr::asr_context::load(mp);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->profile_name(), "qwen3a");

    const std::string audio =
        env_or("ASR_TEST_AUDIO",
               std::string(ASR_SOURCE_DIR) + "/test-data/[P1]她赢得了世界，却输掉了自己【桂冠之下】.wav");

    std::vector<float> pcm;
    ASSERT_TRUE(ctx->load_audio(audio, pcm)) << "could not load audio: " << audio;

    const int    sr  = ctx->sample_rate();
    const size_t cap = (size_t) sr * 15; // 15s clip (1.7B is slower)
    if (pcm.size() > cap) pcm.resize(cap);

    const asr::chunk_text ct = ctx->transcribe_chunk(pcm, asr::transcribe_params{});
    EXPECT_FALSE(ct.text.empty());
    EXPECT_EQ(ct.language, "Chinese");
}
