#include "asr.h"

#include <gtest/gtest.h>

#include <cstring>

// Phase 0 smoke test: verifies the build/test plumbing (CMake + GoogleTest +
// gtest_discover_tests + linking against asr_core) is wired up correctly.
TEST(Smoke, VersionIsNonEmpty) {
    const char * v = asr::version();
    ASSERT_NE(v, nullptr);
    EXPECT_GT(std::strlen(v), 0u);
}
