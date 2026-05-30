#include "asr.h"
#include "asr_profile.h"

#include <gtest/gtest.h>

#include <string>

using asr::select_profile;
using asr::transcribe_params;

TEST(Profile, SelectQwen3a) {
    EXPECT_EQ(select_profile("qwen3a").name, "qwen3a");
}

TEST(Profile, SelectUnknownFallsBackToGeneric) {
    EXPECT_EQ(select_profile("voxtral").name, "generic");
    EXPECT_EQ(select_profile("").name, "generic");
}

TEST(Profile, Qwen3aParseDelegates) {
    auto r = select_profile("qwen3a").parse_output("language English<asr_text>hi there");
    EXPECT_EQ(r.language, "English");
    EXPECT_EQ(r.text, "hi there");
}

TEST(Profile, GenericParseIsPassThrough) {
    auto r = select_profile("anything-else").parse_output("  raw text  ");
    EXPECT_EQ(r.language, ""); // generic never reports a language
    EXPECT_EQ(r.text, "raw text"); // trimmed, otherwise verbatim
}

TEST(Profile, GenericParseKeepsAsrMarkerVerbatim) {
    // Generic must NOT interpret qwen3a's protocol.
    auto r = select_profile("generic").parse_output("language X<asr_text>y");
    EXPECT_EQ(r.language, "");
    EXPECT_EQ(r.text, "language X<asr_text>y");
}

TEST(Profile, BuildPromptMarkerPlusContext) {
    transcribe_params tp;
    tp.context = "cats and dogs";
    EXPECT_EQ(select_profile("qwen3a").build_prompt(tp, "<MARK>"), "<MARK>cats and dogs");
}

TEST(Profile, BuildPromptEmptyContext) {
    transcribe_params tp; // empty context
    EXPECT_EQ(select_profile("qwen3a").build_prompt(tp, "<MARK>"), "<MARK>");
    EXPECT_EQ(select_profile("generic").build_prompt(tp, "<MARK>"), "<MARK>");
}
