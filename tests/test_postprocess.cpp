#include "asr_postprocess.h"

#include <gtest/gtest.h>

#include <string>

using asr::fix_char_repeats;
using asr::fix_pattern_repeats;
using asr::suppress_repeats;

// ---- fix_char_repeats ----

TEST(CharRepeat, NoRepeat) {
    EXPECT_EQ(fix_char_repeats("hello"), "hello");
}

TEST(CharRepeat, UnderThreshold) {
    // 10 'a's, threshold 20 → unchanged.
    EXPECT_EQ(fix_char_repeats(std::string(10, 'a'), 20), std::string(10, 'a'));
}

TEST(CharRepeat, OverThreshold) {
    // 30 'a's, threshold 20 → 20 'a's.
    EXPECT_EQ(fix_char_repeats(std::string(30, 'a'), 20), std::string(20, 'a'));
}

TEST(CharRepeat, CjkRepeat) {
    // "哈哈…" (30 copies of 哈, each 3 bytes), threshold 20.
    std::string s;
    for (int i = 0; i < 30; i++) s += "哈";
    std::string expected;
    for (int i = 0; i < 20; i++) expected += "哈";
    EXPECT_EQ(fix_char_repeats(s, 20), expected);
}

TEST(CharRepeat, MixedContent) {
    // "abc" + 30 'x' + "def" → "abc" + 20 'x' + "def"
    std::string in = "abc" + std::string(30, 'x') + "def";
    std::string expected = "abc" + std::string(20, 'x') + "def";
    EXPECT_EQ(fix_char_repeats(in, 20), expected);
}

// ---- fix_pattern_repeats ----

TEST(PatternRepeat, NoRepeat) {
    EXPECT_EQ(fix_pattern_repeats("hello world"), "hello world");
}

TEST(PatternRepeat, UnderThreshold) {
    // "ab" repeated 5 times, threshold 20 → unchanged.
    std::string s;
    for (int i = 0; i < 5; i++) s += "ab";
    EXPECT_EQ(fix_pattern_repeats(s, 20), s);
}

TEST(PatternRepeat, OverThreshold) {
    // "你好" repeated 25 times, threshold 20 → "你好" (1 copy).
    std::string s;
    for (int i = 0; i < 25; i++) s += "你好";
    EXPECT_EQ(fix_pattern_repeats(s, 20), "你好");
}

TEST(PatternRepeat, ThreeGram) {
    // "abc" repeated 20 times → "abc" (1 copy).
    std::string s;
    for (int i = 0; i < 20; i++) s += "abc";
    EXPECT_EQ(fix_pattern_repeats(s, 20), "abc");
}

TEST(PatternRepeat, PreservesSurrounding) {
    // "X" + "你好"×25 + "Z" → "X" + "你好" + "Z"
    std::string s = "X";
    for (int i = 0; i < 25; i++) s += "你好";
    s += "Z";
    EXPECT_EQ(fix_pattern_repeats(s, 20), "X你好Z");
}

// ---- suppress_repeats (combined) ----

TEST(SuppressRepeats, Combined) {
    // 30 'x' + "你好"×25 → 20 'x' + "你好"
    std::string s = std::string(30, 'x');
    for (int i = 0; i < 25; i++) s += "你好";
    std::string expected = std::string(20, 'x') + "你好";
    EXPECT_EQ(suppress_repeats(s, 20), expected);
}

TEST(SuppressRepeats, Empty) {
    EXPECT_EQ(suppress_repeats(""), "");
}
