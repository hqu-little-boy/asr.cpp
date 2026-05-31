#include "asr_args.h"
#include "asr_error.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

std::expected<asr::cli_args, asr::asr_error> parse(std::vector<const char *> v) {
    return asr::parse_args_checked(static_cast<int>(v.size()), v.data());
}

} // namespace

TEST(Error, MakeErrorCarriesStructuredContext) {
    const auto err = asr::make_error(asr::error_code::io,
                                     "could not read file",
                                     "load_audio",
                                     "missing.wav");
    EXPECT_EQ(err.code, asr::error_code::io);
    EXPECT_EQ(err.stage, "load_audio");
    EXPECT_EQ(err.message, "could not read file");
    EXPECT_EQ(err.path, std::filesystem::path("missing.wav"));
    EXPECT_NE(err.where.file_name(), nullptr);
}

TEST(ArgsChecked, SuccessReturnsValue) {
    auto parsed = parse({"asr-cli", "-m", "model.gguf", "--mmproj", "mmproj.gguf", "audio.wav"});
    ASSERT_TRUE(parsed.has_value()) << parsed.error().message;
    EXPECT_FALSE(parsed->error);
    EXPECT_EQ(parsed->model.model, "model.gguf");
    ASSERT_EQ(parsed->input_files.size(), 1u);
    EXPECT_EQ(parsed->input_files[0], "audio.wav");
}

TEST(ArgsChecked, ParseFailureReturnsStructuredError) {
    auto parsed = parse({"asr-cli", "--mmproj", "mmproj.gguf", "audio.wav"});
    ASSERT_FALSE(parsed.has_value());
    EXPECT_EQ(parsed.error().code, asr::error_code::invalid_argument);
    EXPECT_EQ(parsed.error().stage, "parse_args");
    EXPECT_NE(parsed.error().message.find("model"), std::string::npos);
}

TEST(RunConfig, ConvertsCliPathsToFilesystemPaths) {
    auto parsed = parse({"asr-cli", "-m", "model.gguf", "--mmproj", "mmproj.gguf",
                         "-f", "one.wav", "two.wav", "-p", "3"});
    ASSERT_TRUE(parsed.has_value()) << parsed.error().message;

    const asr::run_config cfg = asr::to_run_config(*parsed);
    EXPECT_EQ(cfg.model.model, "model.gguf");
    EXPECT_EQ(cfg.model.mmproj, "mmproj.gguf");
    EXPECT_EQ(cfg.processors, 3);
    ASSERT_EQ(cfg.input_files.size(), 2u);
    EXPECT_EQ(cfg.input_files[0], std::filesystem::path("one.wav"));
    EXPECT_EQ(cfg.input_files[1], std::filesystem::path("two.wav"));
}
