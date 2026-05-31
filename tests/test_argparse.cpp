#include "asr_args.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using asr::cli_args;
using asr::parse_args;

namespace {
cli_args parse(std::vector<const char *> v) {
    return parse_args(static_cast<int>(v.size()), v.data());
}
} // namespace

TEST(Args, HelpFlag) {
    auto a = parse({"asr-cli", "--help"});
    EXPECT_TRUE(a.help);
    EXPECT_FALSE(a.error);
}

TEST(Args, HelpSkipsRequiredValidation) {
    auto a = parse({"asr-cli", "-h"}); // no model/mmproj/inputs, but help wins
    EXPECT_TRUE(a.help);
    EXPECT_FALSE(a.error);
}

TEST(Args, BasicRequired) {
    auto a = parse({"asr-cli", "-m", "a.gguf", "--mmproj", "p.gguf", "audio.wav"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_EQ(a.model.model, "a.gguf");
    EXPECT_EQ(a.model.mmproj, "p.gguf");
    ASSERT_EQ(a.input_files.size(), 1u);
    EXPECT_EQ(a.input_files[0], "audio.wav");
}

TEST(Args, MissingModel) {
    auto a = parse({"asr-cli", "--mmproj", "p.gguf", "x.wav"});
    EXPECT_TRUE(a.error);
    EXPECT_NE(a.error_msg.find("model"), std::string::npos);
}

TEST(Args, MissingMmproj) {
    auto a = parse({"asr-cli", "-m", "a.gguf", "x.wav"});
    EXPECT_TRUE(a.error);
    EXPECT_NE(a.error_msg.find("mmproj"), std::string::npos);
}

TEST(Args, MissingInput) {
    auto a = parse({"asr-cli", "-m", "a.gguf", "--mmproj", "p.gguf"});
    EXPECT_TRUE(a.error);
    EXPECT_NE(a.error_msg.find("input"), std::string::npos);
}

TEST(Args, OutputFlags) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav",
                    "-otxt", "-oj", "-of", "out/base",
                    "--vad", "--vad-model", "v.gguf"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_TRUE(a.output.out_txt);
    EXPECT_TRUE(a.output.out_json);
    EXPECT_EQ(a.output.out_base, "out/base");
}

TEST(Args, Threads) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "-t", "8"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_EQ(a.model.n_threads, 8);
}

TEST(Args, ProcessorsFlag) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "-p", "4"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_EQ(a.processors, 4);
}

TEST(Args, ProcessorsLongFlag) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "--processors", "16"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_EQ(a.processors, 16);
}

TEST(Args, OutputFormatsRequireVad) {
    // txt alone is fine without --vad
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "-otxt"});
    ASSERT_FALSE(a.error) << a.error_msg;

    // json without --vad should fail
    a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "-oj"});
    EXPECT_TRUE(a.error);
    EXPECT_NE(a.error_msg.find("--vad"), std::string::npos);

    // srt without --vad should fail
    a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "-osrt"});
    EXPECT_TRUE(a.error);

    // json with --vad should succeed
    a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "-oj", "--vad", "--vad-model", "v.gguf"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_TRUE(a.output.out_json);
    EXPECT_TRUE(a.vad.use_vad);
}

TEST(Args, ThreadsBadValue) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "-t", "abc"});
    EXPECT_TRUE(a.error);
}

TEST(Args, NoGpu) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "-ng"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_FALSE(a.model.use_gpu);
    EXPECT_FALSE(a.model.mmproj_use_gpu);
}

TEST(Args, ChunkLength) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "--chunk-length", "25.5"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_FLOAT_EQ(a.transcribe.chunk_length_s, 25.5f);
}

TEST(Args, NPredict) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "-n", "100"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_EQ(a.transcribe.n_predict, 100);
}

TEST(Args, ContextAndProfile) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav",
                    "--context", "foo bar", "--profile", "qwen3a"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_EQ(a.transcribe.context, "foo bar");
    EXPECT_EQ(a.model.profile_override, "qwen3a");
}

TEST(Args, MultipleFilesPositionalAndFlag) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p",
                    "one.wav", "-f", "two.wav", "three.wav"});
    ASSERT_FALSE(a.error) << a.error_msg;
    ASSERT_EQ(a.input_files.size(), 3u);
    EXPECT_EQ(a.input_files[0], "one.wav");
    EXPECT_EQ(a.input_files[1], "two.wav");
    EXPECT_EQ(a.input_files[2], "three.wav");
}

TEST(Args, MissingValueForFlag) {
    auto a = parse({"asr-cli", "-m"});
    EXPECT_TRUE(a.error);
    EXPECT_NE(a.error_msg.find("missing value"), std::string::npos);
}

TEST(Args, UnknownFlag) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "--bogus"});
    EXPECT_TRUE(a.error);
    EXPECT_NE(a.error_msg.find("unknown"), std::string::npos);
}

TEST(Args, DashIsStdinInput) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "-"});
    ASSERT_FALSE(a.error) << a.error_msg;
    ASSERT_EQ(a.input_files.size(), 1u);
    EXPECT_EQ(a.input_files[0], "-");
}

TEST(Args, CarryContextFlag) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "--carry-context"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_TRUE(a.transcribe.carry_context);
}

TEST(Args, SrtVttFlags) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav",
                    "-osrt", "-ovtt", "--vad", "--vad-model", "v.gguf"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_TRUE(a.output.out_srt);
    EXPECT_TRUE(a.output.out_vtt);
}

TEST(Args, VadFlags) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav",
                    "--vad", "--vad-model", "v.gguf",
                    "--vad-threshold", "0.6",
                    "--vad-min-speech", "0.3",
                    "--vad-min-silence", "0.2"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_TRUE(a.vad.use_vad);
    EXPECT_EQ(a.vad.model_path, "v.gguf");
    EXPECT_FLOAT_EQ(a.vad.threshold, 0.6f);
    EXPECT_FLOAT_EQ(a.vad.min_speech_sec, 0.3f);
    EXPECT_FLOAT_EQ(a.vad.min_silence_sec, 0.2f);
}

TEST(Args, VadRequiresModel) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "--vad"});
    EXPECT_TRUE(a.error);
    EXPECT_NE(a.error_msg.find("vad-model"), std::string::npos);
}

TEST(Args, LanguageFlag) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "-l", "Chinese"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_EQ(a.transcribe.language, "Chinese");
}

TEST(Args, LanguageFlagLong) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav", "--language", "English"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_EQ(a.transcribe.language, "English");
}

TEST(Args, SamplingParams) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav",
                    "--temperature", "0.5", "--top-p", "0.9", "--repeat-penalty", "1.2"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_FLOAT_EQ(a.transcribe.temperature, 0.5f);
    EXPECT_FLOAT_EQ(a.transcribe.top_p, 0.9f);
    EXPECT_FLOAT_EQ(a.transcribe.repeat_penalty, 1.2f);
}

TEST(Args, SamplingParamsDefault) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_LT(a.transcribe.temperature, 0.0f); // < 0 = use default
    EXPECT_LT(a.transcribe.top_p, 0.0f);
    EXPECT_LT(a.transcribe.repeat_penalty, 0.0f);
}

TEST(Args, LrcCsvFlags) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav",
                    "-olrc", "-ocsv", "--vad", "--vad-model", "v.gguf"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_TRUE(a.output.out_lrc);
    EXPECT_TRUE(a.output.out_csv);
}

TEST(Args, OutputFormatFlag) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav",
                    "--output-format", "srt", "--output-format", "csv",
                    "--vad", "--vad-model", "v.gguf"});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_TRUE(a.output.out_srt);
    EXPECT_TRUE(a.output.out_csv);
}

TEST(Args, OutputFormatUnknown) {
    auto a = parse({"asr-cli", "-m", "a", "--mmproj", "p", "x.wav",
                    "--output-format", "xyz"});
    EXPECT_TRUE(a.error);
    EXPECT_NE(a.error_msg.find("unknown format"), std::string::npos);
}

TEST(Args, ResponseFile) {
    // Write a temp response file with args.
    const char * path = "/tmp/asr_test_args.txt";
    {
        std::ofstream f(path);
        f << "-m\na.gguf\n# comment\n--mmproj\np.gguf\n\nx.wav\n";
    }
    const std::string at_arg = std::string("@") + path;
    auto a = parse({"asr-cli", at_arg.c_str()});
    ASSERT_FALSE(a.error) << a.error_msg;
    EXPECT_EQ(a.model.model, "a.gguf");
    EXPECT_EQ(a.model.mmproj, "p.gguf");
    ASSERT_EQ(a.input_files.size(), 1u);
    EXPECT_EQ(a.input_files[0], "x.wav");
    std::remove(path);
}

TEST(Args, ResponseFileMissing) {
    // @file that doesn't exist → treated as unknown arg → error.
    auto a = parse({"asr-cli", "@/no/such/file.txt"});
    EXPECT_TRUE(a.error);
}
