#include "asr_job.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

std::filesystem::path unique_base(const char * name) {
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (std::string("asr_job_") + name + "_" + std::to_string(ticks));
}

std::string read_file(const std::filesystem::path & path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

TEST(Job, FinalizeSuppressesRepeatsBeforeWriting) {
    const std::filesystem::path base = unique_base("repeat");

    asr::result r;
    r.text = std::string(25, 'a');
    r.segments.push_back({0, 1000, std::string(25, 'a')});

    asr::output_params out;
    out.out_base = base.string();
    out.out_txt = true;

    auto finalized = asr::finalize_transcription(std::move(r), out, "input.wav");
    ASSERT_TRUE(finalized.has_value()) << finalized.error().message;
    EXPECT_EQ(finalized->transcript.text, std::string(20, 'a'));
    ASSERT_EQ(finalized->transcript.segments.size(), 1u);
    EXPECT_EQ(finalized->transcript.segments[0].text, std::string(20, 'a'));
    EXPECT_EQ(read_file(base.string() + ".txt"), std::string(20, 'a') + "\n");

    std::filesystem::remove(base.string() + ".txt");
}

TEST(Job, FinalizeWithoutOutputFlagsStillReturnsTranscript) {
    asr::result r;
    r.text = "hello";

    asr::output_params out;
    auto finalized = asr::finalize_transcription(std::move(r), out, "input.wav");
    ASSERT_TRUE(finalized.has_value()) << finalized.error().message;
    EXPECT_EQ(finalized->transcript.text, "hello");
    EXPECT_TRUE(finalized->outputs.empty());
}

TEST(Job, FinalizePropagatesWriterError) {
    asr::result r;
    r.text = "hello";

    asr::output_params out;
    out.out_base = (unique_base("missing_parent") / "child").string();
    out.out_txt = true;

    auto finalized = asr::finalize_transcription(std::move(r), out, "input.wav");
    ASSERT_FALSE(finalized.has_value());
    EXPECT_EQ(finalized.error().code, asr::error_code::output);
    EXPECT_EQ(finalized.error().stage, "write_outputs");
}
