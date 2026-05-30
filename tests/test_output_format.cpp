#include "asr_output.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

using asr::json_escape;
using asr::result;
using asr::write_json;
using asr::write_txt;

namespace {
std::string txt_of(const result & r) {
    std::ostringstream os;
    write_txt(os, r);
    return os.str();
}
std::string json_of(const result & r) {
    std::ostringstream os;
    write_json(os, r);
    return os.str();
}
} // namespace

TEST(Output, TxtBasic) {
    result r;
    r.text     = "hello world";
    r.language = "English";
    EXPECT_EQ(txt_of(r), "hello world\n");
}

TEST(Output, TxtEmpty) {
    result r;
    EXPECT_EQ(txt_of(r), "\n");
}

TEST(Output, JsonBasic) {
    result r;
    r.language = "Chinese";
    r.text     = "你好";
    EXPECT_EQ(json_of(r),
              "{\n"
              "  \"language\": \"Chinese\",\n"
              "  \"text\": \"你好\"\n"
              "}\n");
}

TEST(Output, JsonEmptyLanguage) {
    result r;
    r.text = "no language";
    EXPECT_EQ(json_of(r),
              "{\n"
              "  \"language\": \"\",\n"
              "  \"text\": \"no language\"\n"
              "}\n");
}

TEST(Output, JsonEscapingInText) {
    result r;
    r.language = "English";
    r.text     = "a\"b\\c\nd\te";
    EXPECT_EQ(json_of(r),
              "{\n"
              "  \"language\": \"English\",\n"
              "  \"text\": \"a\\\"b\\\\c\\nd\\te\"\n"
              "}\n");
}

TEST(Output, JsonEscapeUnitMandatory) {
    EXPECT_EQ(json_escape("a\"b"), "a\\\"b");
    EXPECT_EQ(json_escape("a\\b"), "a\\\\b");
    EXPECT_EQ(json_escape("a\nb"), "a\\nb");
    EXPECT_EQ(json_escape("a\rb"), "a\\rb");
    EXPECT_EQ(json_escape("a\tb"), "a\\tb");
}

TEST(Output, JsonEscapeUnitControlChar) {
    EXPECT_EQ(json_escape(std::string("a\x01z")), "a\\u0001z");
}

TEST(Output, JsonEscapeUnitUtf8Verbatim) {
    // CJK bytes must be passed through, not \u-escaped.
    EXPECT_EQ(json_escape("你好世界"), "你好世界");
}
