#include "asr_engine.h"

#include <gtest/gtest.h>

#include <span>
#include <type_traits>
#include <utility>
#include <vector>

TEST(EngineApi, TranscribeChunkAcceptsSpanAndVector) {
    using SpanReturn = decltype(std::declval<asr::asr_context &>().transcribe_chunk(
        std::declval<std::span<const float>>(),
        std::declval<const asr::transcribe_params &>()));
    using VectorReturn = decltype(std::declval<asr::asr_context &>().transcribe_chunk(
        std::declval<const std::vector<float> &>(),
        std::declval<const asr::transcribe_params &>()));

    static_assert(std::is_same_v<SpanReturn, asr::chunk_text>);
    static_assert(std::is_same_v<VectorReturn, asr::chunk_text>);
    SUCCEED();
}
