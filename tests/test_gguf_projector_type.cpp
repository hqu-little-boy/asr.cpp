#include "asr_gguf.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#ifndef ASR_SOURCE_DIR
#define ASR_SOURCE_DIR "."
#endif

// Always runs: no model required.
TEST(Gguf, MissingFileReturnsEmpty) {
    EXPECT_EQ(asr::read_projector_type("/no/such/file.gguf"), "");
}

// Reads the real mmproj; gated behind ASR_RUN_MODEL_TESTS.
TEST(Gguf, ReadsQwen3aProjectorType) {
    if (!std::getenv("ASR_RUN_MODEL_TESTS")) {
        GTEST_SKIP() << "set ASR_RUN_MODEL_TESTS=1 to run model-dependent tests";
    }
    const std::string mmproj = std::string(ASR_SOURCE_DIR) + "/models/mmproj-Qwen3-ASR-0.6B-bf16.gguf";
    EXPECT_EQ(asr::read_projector_type(mmproj), "qwen3a");
}
