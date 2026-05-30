#include "asr_profile.h"

#include <gtest/gtest.h>

using asr::parse_qwen3a_output;

TEST(ParseQwen3a, Normal) {
    auto r = parse_qwen3a_output("language Chinese<asr_text>我在五点了，是睡觉。");
    EXPECT_EQ(r.language, "Chinese");
    EXPECT_EQ(r.text, "我在五点了，是睡觉。");
}

TEST(ParseQwen3a, EnglishLanguage) {
    auto r = parse_qwen3a_output("language English<asr_text>hello world");
    EXPECT_EQ(r.language, "English");
    EXPECT_EQ(r.text, "hello world");
}

TEST(ParseQwen3a, NoPreamble) {
    auto r = parse_qwen3a_output("<asr_text>hello");
    EXPECT_EQ(r.language, "");
    EXPECT_EQ(r.text, "hello");
}

TEST(ParseQwen3a, NoMarkerWholeIsText) {
    auto r = parse_qwen3a_output("just some text");
    EXPECT_EQ(r.language, "");
    EXPECT_EQ(r.text, "just some text");
}

TEST(ParseQwen3a, EmptyInput) {
    auto r = parse_qwen3a_output("");
    EXPECT_EQ(r.language, "");
    EXPECT_EQ(r.text, "");
}

TEST(ParseQwen3a, TextContainsAngleBrackets) {
    auto r = parse_qwen3a_output("language English<asr_text>a < b and c > d");
    EXPECT_EQ(r.language, "English");
    EXPECT_EQ(r.text, "a < b and c > d");
}

TEST(ParseQwen3a, MultipleMarkersTakeFirst) {
    auto r = parse_qwen3a_output("language X<asr_text>foo<asr_text>bar");
    EXPECT_EQ(r.language, "X");
    EXPECT_EQ(r.text, "foo<asr_text>bar");
}

TEST(ParseQwen3a, CaseInsensitiveKeyword) {
    auto r = parse_qwen3a_output("Language Chinese<asr_text>x");
    EXPECT_EQ(r.language, "Chinese");
    EXPECT_EQ(r.text, "x");
}

TEST(ParseQwen3a, EmptyTranscriptionAfterMarker) {
    auto r = parse_qwen3a_output("language Chinese<asr_text>");
    EXPECT_EQ(r.language, "Chinese");
    EXPECT_EQ(r.text, "");
}

TEST(ParseQwen3a, LeadingTrailingWhitespaceTrimmed) {
    auto r = parse_qwen3a_output("language Chinese<asr_text>  hi there  ");
    EXPECT_EQ(r.language, "Chinese");
    EXPECT_EQ(r.text, "hi there");
}

TEST(ParseQwen3a, MultiWordLanguageValue) {
    auto r = parse_qwen3a_output("language Simplified Chinese<asr_text>x");
    EXPECT_EQ(r.language, "Simplified Chinese");
    EXPECT_EQ(r.text, "x");
}

// "language" keyword with no following value must not crash or misparse.
TEST(ParseQwen3a, KeywordWithoutValue) {
    auto r = parse_qwen3a_output("language<asr_text>x");
    EXPECT_EQ(r.language, "");
    EXPECT_EQ(r.text, "x");
}

// A preamble that is not a "language ..." line is discarded, not emitted as text.
TEST(ParseQwen3a, NonLanguagePreambleDiscarded) {
    auto r = parse_qwen3a_output("noise here<asr_text>real text");
    EXPECT_EQ(r.language, "");
    EXPECT_EQ(r.text, "real text");
}
