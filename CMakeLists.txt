cmake_minimum_required(VERSION 3.18)
project(asr_cpp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ASR_BUILD_ENGINE "Build the mtmd-backed engine and full CLI (requires llama.cpp)" ON)
option(ASR_BUILD_TESTS  "Build unit tests (GoogleTest)" ON)

# ---------------------------------------------------------------------------
# asr_core: pure-logic library (NO llama.cpp / mtmd dependency).
# Holds parsing, chunking, formatting, argument parsing, profile registry and
# merging, so the bulk of the logic is unit-testable without a model.
# ---------------------------------------------------------------------------
add_library(asr_core STATIC
    src/asr_types.cpp
    src/asr_profile.cpp
)
target_include_directories(asr_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(asr_core PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# asr-cli: command-line front-end (thin shell over the library).
# ---------------------------------------------------------------------------
add_executable(asr-cli cli/asr_cli.cpp)
target_link_libraries(asr-cli PRIVATE asr_core)

# ---------------------------------------------------------------------------
# asr_engine: mtmd-backed engine (optional; wired up in Phase 2).
# When enabled, llama.cpp is built from source (common + the mtmd library);
# examples / server / tools stay OFF because LLAMA_STANDALONE is OFF here.
# ---------------------------------------------------------------------------
if(ASR_BUILD_ENGINE)
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
    add_subdirectory(llama.cpp)
    add_subdirectory(llama.cpp/tools/mtmd ${CMAKE_BINARY_DIR}/mtmd-build)
    # asr_engine library + CLI transcription wiring are added in Phase 2.
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(ASR_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    include(GoogleTest)

    add_executable(test_smoke tests/test_smoke.cpp)
    target_link_libraries(test_smoke PRIVATE asr_core GTest::gtest_main)
    gtest_discover_tests(test_smoke)
endif()
