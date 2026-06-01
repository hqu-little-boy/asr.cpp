#ifdef _MSC_VER
#  define _USE_MATH_DEFINES  // make M_PI available on MSVC
#endif
#include <cmath>

#include "asr_chunker.h"

#include <gtest/gtest.h>

#include <vector>

using asr::audio_chunk;
using asr::chunk_audio;

namespace {

constexpr int kSR = 16000;

// Append `n` samples of a sine tone (amplitude 0.5) to `out`.
void append_tone(std::vector<float> & out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out.push_back(0.5f * std::sin(2.0 * M_PI * 440.0 * (double) i / kSR));
    }
}

// Append `n` samples of silence (zeros).
void append_silence(std::vector<float> & out, size_t n) {
    out.insert(out.end(), n, 0.0f);
}

// Verify chunks tile [0, n) exactly with no gaps or overlaps.
void expect_full_coverage(const std::vector<audio_chunk> & chunks, size_t n) {
    ASSERT_FALSE(chunks.empty());
    EXPECT_EQ(chunks.front().offset, 0u);
    size_t pos = 0;
    for (const auto & c : chunks) {
        EXPECT_EQ(c.offset, pos);
        EXPECT_GT(c.length, 0u);
        pos += c.length;
    }
    EXPECT_EQ(pos, n);
}

} // namespace

TEST(Chunker, EmptyInputNoChunks) {
    std::vector<float> pcm;
    auto chunks = chunk_audio(pcm.data(), pcm.size(), kSR, 3.0f);
    EXPECT_TRUE(chunks.empty());
}

TEST(Chunker, ShortAudioSingleChunk) {
    std::vector<float> pcm;
    append_tone(pcm, kSR / 2); // 0.5 s, shorter than a 3 s chunk
    auto chunks = chunk_audio(pcm.data(), pcm.size(), kSR, 3.0f);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].offset, 0u);
    EXPECT_EQ(chunks[0].length, pcm.size());
}

TEST(Chunker, ZeroChunkLengthSingleChunk) {
    std::vector<float> pcm;
    append_tone(pcm, kSR * 10);
    auto chunks = chunk_audio(pcm.data(), pcm.size(), kSR, 0.0f);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].length, pcm.size());
}

TEST(Chunker, UniformToneCutsAtTarget) {
    // 6 s of constant-energy tone, no silence -> cut should stay at the target.
    std::vector<float> pcm;
    append_tone(pcm, kSR * 6);
    const size_t target = (size_t) (3.0f * kSR);
    auto chunks = chunk_audio(pcm.data(), pcm.size(), kSR, 3.0f, /*search_s=*/0.5f);
    ASSERT_GE(chunks.size(), 2u);
    EXPECT_EQ(chunks[0].length, target);
    expect_full_coverage(chunks, pcm.size());
}

TEST(Chunker, CutsAtSilenceNearTarget) {
    // tone | 0.2 s silence centered at 3 s | tone  -> first cut lands in silence.
    const size_t pre   = (size_t) (2.9f * kSR);   // tone up to 2.9 s
    const size_t sil   = (size_t) (0.2f * kSR);   // silence [2.9 s, 3.1 s)
    const size_t post  = (size_t) (3.0f * kSR);   // tone afterwards
    std::vector<float> pcm;
    append_tone(pcm, pre);
    append_silence(pcm, sil);
    append_tone(pcm, post);

    auto chunks = chunk_audio(pcm.data(), pcm.size(), kSR, 3.0f, /*search_s=*/0.5f);
    ASSERT_GE(chunks.size(), 2u);
    // First chunk should end within the silence region [pre, pre+sil].
    const size_t cut = chunks[0].length;
    EXPECT_GE(cut, pre);
    EXPECT_LE(cut, pre + sil);
    expect_full_coverage(chunks, pcm.size());
}

TEST(Chunker, ManyChunksFullCoverage) {
    std::vector<float> pcm;
    append_tone(pcm, kSR * 17); // 17 s -> several 3 s chunks
    auto chunks = chunk_audio(pcm.data(), pcm.size(), kSR, 3.0f, /*search_s=*/0.5f);
    ASSERT_GE(chunks.size(), 5u);
    expect_full_coverage(chunks, pcm.size());
}
