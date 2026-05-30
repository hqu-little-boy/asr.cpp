#include "asr_merge.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using asr::audio_chunk;
using asr::chunk_result;
using asr::chunk_text;
using asr::merge_chunks;

namespace {
constexpr int kSR = 16000;
chunk_result cr(size_t off, size_t len, std::string text, std::string lang = "") {
    return chunk_result{audio_chunk{off, len}, chunk_text{std::move(text), std::move(lang)}};
}
} // namespace

TEST(Merge, EmptyInput) {
    auto r = merge_chunks({}, kSR);
    EXPECT_EQ(r.text, "");
    EXPECT_TRUE(r.segments.empty());
    EXPECT_EQ(r.language, "");
}

TEST(Merge, SingleChunk) {
    auto r = merge_chunks({cr(0, kSR, "hello", "English")}, kSR);
    EXPECT_EQ(r.text, "hello");
    EXPECT_EQ(r.language, "English");
    ASSERT_EQ(r.segments.size(), 1u);
    EXPECT_EQ(r.segments[0].t0_ms, 0);
    EXPECT_EQ(r.segments[0].t1_ms, 1000);
    EXPECT_EQ(r.segments[0].text, "hello");
}

TEST(Merge, EnglishChunksJoinedWithSpace) {
    auto r = merge_chunks({cr(0, kSR, "hello world"), cr(kSR, kSR, "foo bar")}, kSR);
    EXPECT_EQ(r.text, "hello world foo bar");
    EXPECT_EQ(r.segments.size(), 2u);
}

TEST(Merge, CjkChunksNotSplitBySpace) {
    auto r = merge_chunks({cr(0, kSR, "你好"), cr(kSR, kSR, "世界")}, kSR);
    EXPECT_EQ(r.text, "你好世界");
}

TEST(Merge, EmptyTextChunksSkipped) {
    auto r = merge_chunks({cr(0, kSR, "abc"), cr(kSR, kSR, ""), cr(2 * kSR, kSR, "def")}, kSR);
    EXPECT_EQ(r.text, "abc def");
    EXPECT_EQ(r.segments.size(), 2u);
}

TEST(Merge, LanguageFromFirstNonEmpty) {
    auto r = merge_chunks({cr(0, kSR, "a", ""), cr(kSR, kSR, "b", "English"),
                           cr(2 * kSR, kSR, "c", "French")}, kSR);
    EXPECT_EQ(r.language, "English");
}

TEST(Merge, LanguageTakenEvenWhenThatChunkTextEmpty) {
    auto r = merge_chunks({cr(0, kSR, "", "Chinese"), cr(kSR, kSR, "hi", "English")}, kSR);
    EXPECT_EQ(r.language, "Chinese");
    EXPECT_EQ(r.text, "hi");
    ASSERT_EQ(r.segments.size(), 1u);
    EXPECT_EQ(r.segments[0].text, "hi");
}

TEST(Merge, SegmentTimesFromSpan) {
    auto r = merge_chunks({cr(kSR, 2 * kSR, "x")}, kSR); // offset 1s, length 2s
    ASSERT_EQ(r.segments.size(), 1u);
    EXPECT_EQ(r.segments[0].t0_ms, 1000);
    EXPECT_EQ(r.segments[0].t1_ms, 3000);
}
