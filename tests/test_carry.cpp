#include "asr_carry.h"

#include <gtest/gtest.h>

#include <string>

using asr::carry_context;

TEST(Carry, EmptyPrevReturnsBase) {
    EXPECT_EQ(carry_context("", "hotwords"), "hotwords");
    EXPECT_EQ(carry_context("", ""), "");
}

TEST(Carry, ShortPrevAppendedToBaseWithSpace) {
    EXPECT_EQ(carry_context("hello", "ctx"), "ctx hello");
}

TEST(Carry, EmptyBaseReturnsTailOnly) {
    EXPECT_EQ(carry_context("hello world", ""), "hello world");
}

TEST(Carry, TailTruncatedToMaxChars) {
    const std::string prev(10, 'a');
    EXPECT_EQ(carry_context(prev, "", 5), "aaaaa");
}

TEST(Carry, TailDoesNotSplitUtf8) {
    // "你好吗" is 9 bytes (3 per char). max_chars=4 lands mid-character; the
    // tail must advance to a lead byte, yielding the final whole char "吗".
    EXPECT_EQ(carry_context("你好吗", "", 4), "吗");
}

TEST(Carry, BaseAndLongTailJoinedWithSpace) {
    const std::string prev(300, 'x');
    EXPECT_EQ(carry_context(prev, "bias", 100), "bias " + std::string(100, 'x'));
}
