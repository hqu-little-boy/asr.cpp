// test_dedup.cpp — Tests for merge dedup (overlap at chunk boundaries) and
// inter-segment dedup_segments.

#include "asr_merge.h"
#include "asr_postprocess.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using asr::audio_chunk;
using asr::chunk_result;
using asr::chunk_text;
using asr::dedup_segments;
using asr::merge_chunks;
using asr::result;
using asr::segment;

namespace {
constexpr int kSR = 16000;

chunk_result cr(size_t off, size_t len, std::string text, std::string lang = "") {
    return chunk_result{audio_chunk{off, len}, chunk_text{std::move(text), std::move(lang)}};
}
} // namespace

// ---- Merge dedup (overlap at chunk boundaries) ----

TEST(MergeDedup, NoOverlap) {
    auto r = merge_chunks({cr(0, kSR, "hello world"), cr(kSR, kSR, "foo bar")}, kSR);
    EXPECT_EQ(r.segments.size(), 2u);
    EXPECT_EQ(r.segments[0].text, "hello world");
    EXPECT_EQ(r.segments[1].text, "foo bar");
}

TEST(MergeDedup, OverlapTrimmed) {
    // Chunk 1 ends with "the quick brown fox", chunk 2 starts with "brown fox jumps".
    // "brown fox" (9 chars) overlaps → trimmed from chunk 2.
    auto r = merge_chunks({
        cr(0, kSR, "the quick brown fox"),
        cr(kSR, kSR, "brown fox jumps over")
    }, kSR);
    ASSERT_EQ(r.segments.size(), 2u);
    EXPECT_EQ(r.segments[0].text, "the quick brown fox");
    EXPECT_EQ(r.segments[1].text, "jumps over");
}

TEST(MergeDedup, FullDuplicateSkipped) {
    // Chunk 2 is entirely contained in chunk 1 → skipped.
    auto r = merge_chunks({
        cr(0, kSR, "hello world"),
        cr(kSR, kSR, "hello world")
    }, kSR);
    ASSERT_EQ(r.segments.size(), 1u);
    EXPECT_EQ(r.segments[0].text, "hello world");
}

TEST(MergeDedup, ShortOverlapIgnored) {
    // Overlap < 8 chars → not trimmed (too short, likely coincidental).
    auto r = merge_chunks({
        cr(0, kSR, "abc def"),
        cr(kSR, kSR, "def ghi")
    }, kSR);
    ASSERT_EQ(r.segments.size(), 2u);
    EXPECT_EQ(r.segments[1].text, "def ghi"); // "def" (3 chars) < min_overlap=8
}

TEST(MergeDedup, CjkOverlap) {
    // CJK overlap: "你好世界再见" ends with "世界再见", chunk 2 starts with "世界再见朋友".
    auto r = merge_chunks({
        cr(0, kSR, "你好世界再见"),
        cr(kSR, kSR, "世界再见朋友")
    }, kSR);
    ASSERT_EQ(r.segments.size(), 2u);
    EXPECT_EQ(r.segments[0].text, "你好世界再见");
    EXPECT_EQ(r.segments[1].text, "朋友");
}

// ---- Inter-segment dedup_segments ----

TEST(SegDedup, EmptyResult) {
    result r;
    dedup_segments(r);
    EXPECT_TRUE(r.segments.empty());
}

TEST(SegDedup, SingleSegment) {
    result r;
    r.segments = {{0, 1000, "hello"}};
    dedup_segments(r);
    EXPECT_EQ(r.segments.size(), 1u);
}

TEST(SegDedup, NoDuplication) {
    result r;
    r.segments = {{0, 1000, "hello"}, {1000, 2000, "world"}};
    dedup_segments(r);
    EXPECT_EQ(r.segments.size(), 2u);
}

TEST(SegDedup, PrefixDupKeepsLonger) {
    // Segment 2 starts with segment 1's text (and is longer) → drop segment 1.
    // "你好世界" is 4 codepoints → pass min_common=3 to trigger dedup.
    result r;
    r.segments = {{0, 1000, "你好世界"}, {1000, 2000, "你好世界再见"}};
    dedup_segments(r, 3);
    ASSERT_EQ(r.segments.size(), 1u);
    EXPECT_EQ(r.segments[0].text, "你好世界再见");
}

TEST(SegDedup, PrefixDupKeepsLongerReverse) {
    // Segment 1 starts with segment 2's text (and is longer) → drop segment 2.
    result r;
    r.segments = {{0, 1000, "你好世界再见"}, {1000, 2000, "你好世界"}};
    dedup_segments(r, 3);
    ASSERT_EQ(r.segments.size(), 1u);
    EXPECT_EQ(r.segments[0].text, "你好世界再见");
}

TEST(SegDedup, ShortPrefixNotDeduped) {
    // Common prefix (5 codepoints) < min_common=8 → not deduped.
    result r;
    r.segments = {{0, 1000, "hello world"}, {1000, 2000, "hello there"}};
    dedup_segments(r); // default min_common=8
    EXPECT_EQ(r.segments.size(), 2u);
}

TEST(SegDedup, TextRebuiltAfterDedup) {
    result r;
    r.segments = {{0, 1000, "abc def"}, {1000, 2000, "abc def ghi"}};
    dedup_segments(r, 3);
    ASSERT_EQ(r.segments.size(), 1u);
    EXPECT_EQ(r.text, "abc def ghi");
}
