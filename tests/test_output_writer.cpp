#include "asr_output_writer.h"

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
           (std::string("asr_") + name + "_" + std::to_string(ticks));
}

std::string read_file(const std::filesystem::path & path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

asr::result sample_result() {
    asr::result r;
    r.language = "English";
    r.text = "hello world";
    r.segments.push_back({0, 1000, "hello world"});
    return r;
}

} // namespace

TEST(OutputWriter, SelectedFormatsPreserveCliOrder) {
    asr::output_params p;
    p.out_json = true;
    p.out_txt  = true;
    p.out_csv  = true;

    const auto formats = asr::selected_output_formats(p);
    ASSERT_EQ(formats.size(), 3u);
    EXPECT_EQ(formats[0], asr::output_format::txt);
    EXPECT_EQ(formats[1], asr::output_format::json);
    EXPECT_EQ(formats[2], asr::output_format::csv);
}

TEST(OutputWriter, WritesTxtAndJsonFiles) {
    const std::filesystem::path base = unique_base("txt_json");

    asr::output_params p;
    p.out_base = base.string();
    p.out_txt = true;
    p.out_json = true;

    auto written = asr::write_selected_outputs(sample_result(), p, "input.wav");
    ASSERT_TRUE(written.has_value()) << written.error().message;
    ASSERT_EQ(written->size(), 2u);
    EXPECT_EQ((*written)[0].path, base.string() + ".txt");
    EXPECT_EQ((*written)[1].path, base.string() + ".json");
    EXPECT_EQ(read_file(base.string() + ".txt"), "hello world\n");
    EXPECT_NE(read_file(base.string() + ".json").find("\"language\": \"English\""), std::string::npos);

    std::filesystem::remove(base.string() + ".txt");
    std::filesystem::remove(base.string() + ".json");
}

TEST(OutputWriter, CueFormatsReportCueCount) {
    const std::filesystem::path base = unique_base("srt");

    asr::output_params p;
    p.out_base = base.string();
    p.out_srt = true;

    auto written = asr::write_selected_outputs(sample_result(), p, "input.wav");
    ASSERT_TRUE(written.has_value()) << written.error().message;
    ASSERT_EQ(written->size(), 1u);
    EXPECT_EQ((*written)[0].format, asr::output_format::srt);
    EXPECT_EQ((*written)[0].cue_count, 1u);
    EXPECT_NE(read_file(base.string() + ".srt").find("hello world"), std::string::npos);

    std::filesystem::remove(base.string() + ".srt");
}

TEST(OutputWriter, WriteFailureReturnsStructuredError) {
    asr::output_params p;
    p.out_base = (unique_base("missing_parent") / "child").string();
    p.out_txt = true;

    auto written = asr::write_selected_outputs(sample_result(), p, "input.wav");
    ASSERT_FALSE(written.has_value());
    EXPECT_EQ(written.error().code, asr::error_code::output);
    EXPECT_EQ(written.error().stage, "write_outputs");
    EXPECT_NE(written.error().message.find("cannot write"), std::string::npos);
}
