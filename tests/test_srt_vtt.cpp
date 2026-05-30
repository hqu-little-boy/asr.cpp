#include "asr.h"
#include "asr_output.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

using asr::cue_params;
using asr::result;
using asr::segment;
using asr::split_cues;
using asr::subtitle_cue;
using asr::to_srt_timestamp;
using asr::to_vtt_timestamp;
using asr::write_csv;
using asr::write_lrc;
using asr::write_srt;
using asr::write_vtt;

// ---- Timestamp formatting ----

TEST(SrtTimestamp, Zero) {
    EXPECT_EQ(to_srt_timestamp(0), "00:00:00,000");
}

TEST(SrtTimestamp, Basic) {
    EXPECT_EQ(to_srt_timestamp(3661001), "01:01:01,001");
}

TEST(SrtTimestamp, MaxReasonable) {
    EXPECT_EQ(to_srt_timestamp(35999999), "09:59:59,999");
}

TEST(VttTimestamp, UsesDot) {
    EXPECT_EQ(to_vtt_timestamp(1500), "00:00:01.500");
}

// ---- SRT / VTT writers ----

TEST(SrtWriter, Empty) {
    std::ostringstream os;
    write_srt(os, {});
    EXPECT_EQ(os.str(), "");
}

TEST(SrtWriter, OneCue) {
    std::vector<subtitle_cue> cues = {{1, 0, 2500, "hello world"}};
    std::ostringstream os;
    write_srt(os, cues);
    EXPECT_EQ(os.str(),
              "1\n"
              "00:00:00,000 --> 00:00:02,500\n"
              "hello world\n\n");
}

TEST(SrtWriter, TwoCues) {
    std::vector<subtitle_cue> cues = {{1, 0, 1000, "a"}, {2, 1000, 2000, "b"}};
    std::ostringstream os;
    write_srt(os, cues);
    EXPECT_EQ(os.str(),
              "1\n00:00:00,000 --> 00:00:01,000\na\n\n"
              "2\n00:00:01,000 --> 00:00:02,000\nb\n\n");
}

TEST(VttWriter, Header) {
    std::vector<subtitle_cue> cues = {{1, 0, 1000, "x"}};
    std::ostringstream os;
    write_vtt(os, cues);
    EXPECT_EQ(os.str(),
              "WEBVTT\n\n"
              "00:00:00.000 --> 00:00:01.000\nx\n\n");
}

// ---- Cue splitting ----

TEST(CueSplit, NoSegmentsEmptyResult) {
    result r;
    auto cues = split_cues(r);
    EXPECT_TRUE(cues.empty());
}

TEST(CueSplit, ShortTextOneCue) {
    result r;
    r.segments = {{0, 2000, "hello"}};
    auto cues = split_cues(r);
    ASSERT_EQ(cues.size(), 1u);
    EXPECT_EQ(cues[0].text, "hello");
    EXPECT_EQ(cues[0].start_ms, 0);
    EXPECT_EQ(cues[0].end_ms, 2000);
    EXPECT_EQ(cues[0].index, 1);
}

TEST(CueSplit, SplitOnChinesePeriod) {
    result r;
    // "你好世界。再见世界。" — two sentences split by 。 (10 codepoints total).
    r.segments = {{0, 4000, "你好世界。再见世界。"}};
    cue_params p;
    p.max_chars = 6; // 10 > 6, triggers split at sentence-ending 。
    auto cues = split_cues(r, p);
    ASSERT_EQ(cues.size(), 2u);
    EXPECT_EQ(cues[0].text, "你好世界。");
    EXPECT_EQ(cues[1].text, "再见世界。");
}

TEST(CueSplit, SplitOnExclamationAndQuestion) {
    result r;
    r.segments = {{0, 3000, "你好！再见？"}}; // 6 codepoints
    cue_params p;
    p.max_chars = 4; // 6 > 4, triggers split at ！
    auto cues = split_cues(r, p);
    ASSERT_EQ(cues.size(), 2u);
    EXPECT_EQ(cues[0].text, "你好！");
    EXPECT_EQ(cues[1].text, "再见？");
}

TEST(CueSplit, LongTextForceSplit) {
    result r;
    // 20 codepoints, no punctuation → force-split at max_chars=10.
    r.segments = {{0, 4000, "一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾"}};
    cue_params p;
    p.max_chars = 10;
    auto cues = split_cues(r, p);
    ASSERT_EQ(cues.size(), 2u);
    EXPECT_EQ(cues[0].text, "一二三四五六七八九十");
    EXPECT_EQ(cues[1].text, "壹贰叁肆伍陆柒捌玖拾");
}

TEST(CueSplit, InterpolatedTimes) {
    result r;
    // "AB。CD" — 4 codepoints, split at period after 2 → each half gets ~half the time.
    r.segments = {{0, 4000, "AB。CD"}};
    cue_params p;
    p.max_chars = 3; // force split
    auto cues = split_cues(r, p);
    ASSERT_EQ(cues.size(), 2u);
    EXPECT_EQ(cues[0].start_ms, 0);
    // The split happens at byte offset of the 。 (byte 2 in ASCII "AB.CD" is fine;
    // but for CJK, "AB。CD" is bytes 0-1="AB", 2-4="。", 5-6="CD", total 7 bytes).
    EXPECT_EQ(cues[1].end_ms, 4000);
}

TEST(CueSplit, CjkNoSpuriousSpaces) {
    result r;
    r.segments = {{0, 2000, "你好世界"}};
    cue_params p;
    p.max_chars = 100; // no split needed
    auto cues = split_cues(r, p);
    ASSERT_EQ(cues.size(), 1u);
    EXPECT_EQ(cues[0].text, "你好世界");
}

TEST(CueSplit, AsciiSplitOnPeriod) {
    result r;
    // "Hello. World." — split at the first '.' (after "Hello", byte offset 5).
    r.segments = {{0, 3000, "Hello. World."}};
    cue_params p;
    p.max_chars = 8;
    auto cues = split_cues(r, p);
    ASSERT_EQ(cues.size(), 2u);
    EXPECT_EQ(cues[0].text, "Hello.");   // includes the period
    EXPECT_EQ(cues[1].text, " World.");  // includes the leading space
}

TEST(CueSplit, IndexIncrements) {
    result r;
    // "a。b。c" — 5 codepoints, split at sentence-ending 。.
    r.segments = {{0, 2000, "a。b。c"}};
    cue_params p;
    p.max_chars = 3; // 5 > 3, triggers split
    auto cues = split_cues(r, p);
    ASSERT_EQ(cues.size(), 3u);
    EXPECT_EQ(cues[0].index, 1);
    EXPECT_EQ(cues[1].index, 2);
    EXPECT_EQ(cues[2].index, 3);
}

TEST(CueSplit, MultiSegmentIndicesCarry) {
    result r;
    r.segments = {{0, 2000, "a。b"}, {2000, 4000, "c。d"}};
    cue_params p;
    p.max_chars = 2;
    auto cues = split_cues(r, p);
    ASSERT_EQ(cues.size(), 4u);
    EXPECT_EQ(cues[0].index, 1);
    EXPECT_EQ(cues[2].index, 3); // second segment continues numbering
}

// ---- LRC writer ----

TEST(LrcWriter, Basic) {
    std::vector<subtitle_cue> cues = {{1, 0, 2500, "hello"}, {2, 2500, 5000, "world"}};
    std::ostringstream os;
    write_lrc(os, cues);
    EXPECT_EQ(os.str(), "[00:00.00]hello\n[00:02.50]world\n");
}

TEST(LrcWriter, Empty) {
    std::ostringstream os;
    write_lrc(os, {});
    EXPECT_EQ(os.str(), "");
}

// ---- CSV writer ----

TEST(CsvWriter, Basic) {
    std::vector<subtitle_cue> cues = {{1, 0, 2500, "hello"}, {2, 2500, 5000, "world"}};
    std::ostringstream os;
    write_csv(os, cues);
    EXPECT_EQ(os.str(),
              "start,end,text\n"
              "0,2.5,\"hello\"\n"
              "2.5,5,\"world\"\n");
}

TEST(CsvWriter, Escaping) {
    std::vector<subtitle_cue> cues = {{1, 0, 1000, "a\"b\nc"}};
    std::ostringstream os;
    write_csv(os, cues);
    EXPECT_NE(os.str().find("\"a\\\"b\\nc\""), std::string::npos);
}

TEST(CsvWriter, Empty) {
    std::ostringstream os;
    write_csv(os, {});
    EXPECT_EQ(os.str(), "start,end,text\n");
}
