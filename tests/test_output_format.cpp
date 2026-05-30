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

TEST(Output, JsonFullBasic) {
    result r;
    r.language = "Chinese";
    r.text     = "你好世界。再见。";
    r.segments = {{0, 2000, "你好世界。"}, {2000, 4000, "再见。"}};
    std::ostringstream os;
    write_json_full(os, r);
    const std::string expected =
        "{\n"
        "  \"language\": \"Chinese\",\n"
        "  \"text\": \"你好世界。再见。\",\n"
        "  \"segments\": [\n"
        "    {\"start\": 0, \"end\": 2, \"text\": \"你好世界。\"},\n"
        "    {\"start\": 2, \"end\": 4, \"text\": \"再见。\"}\n"
        "  ]\n"
        "}\n";
    EXPECT_EQ(os.str(), expected);
}

TEST(Output, JsonFullSubsecond) {
    result r;
    r.language = "English";
    r.text     = "hi";
    r.segments = {{149, 1529, "hi"}};
    std::ostringstream os;
    write_json_full(os, r);
    EXPECT_NE(os.str().find("\"start\": 0.149"), std::string::npos);
    EXPECT_NE(os.str().find("\"end\": 1.529"), std::string::npos);
}

TEST(Output, JsonFullEmptySegments) {
    result r;
    r.language = "";
    r.text     = "";
    std::ostringstream os;
    write_json_full(os, r);
    // Empty segments array still has newline-separated brackets.
    EXPECT_NE(os.str().find("\"segments\": ["), std::string::npos);
    EXPECT_NE(os.str().find("]"), std::string::npos);
}
