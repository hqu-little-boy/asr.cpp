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

find_package(CLI11 REQUIRED)

# ---------------------------------------------------------------------------
# asr_core: pure-logic library (NO llama.cpp / mtmd dependency).
# Holds parsing, chunking, formatting, argument parsing, profile registry and
# merging, so the bulk of the logic is unit-testable without a model.
# ---------------------------------------------------------------------------
add_library(asr_core STATIC
    src/asr_types.cpp
    src/asr_profile.cpp
    src/asr_chunker.cpp
    src/asr_output.cpp
    src/asr_args.cpp
    src/asr_merge.cpp
    src/asr_carry.cpp
    src/asr_postprocess.cpp
)
target_include_directories(asr_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(asr_core PUBLIC cxx_std_17)
target_link_libraries(asr_core PRIVATE CLI11::CLI11)

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
    # tools/mtmd is added directly (not via llama.cpp's own tools/ subdir), so it
    # does not inherit LLAMA_INSTALL_VERSION from llama.cpp's subdirectory scope.
    # Define it here so mtmd's set_target_properties(VERSION ...) is well-formed.
    set(LLAMA_INSTALL_VERSION 0.0.0)
    add_subdirectory(llama.cpp/tools/mtmd ${CMAKE_BINARY_DIR}/mtmd-build)

    # FireRedVAD (DFSMN) voice-activity detector — links only ggml (for gguf).
    add_library(asr_vad STATIC src/asr_vad.cpp)
    target_include_directories(asr_vad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(asr_vad PUBLIC ggml)
    target_compile_features(asr_vad PRIVATE cxx_std_17)

    # mtmd-backed engine library.
    add_library(asr_engine STATIC
        src/asr_gguf.cpp
        src/asr_audio.cpp
        src/asr_engine.cpp
        src/asr_driver.cpp
    )
    target_include_directories(asr_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(asr_engine PUBLIC asr_core mtmd llama-common)
    target_compile_features(asr_engine PRIVATE cxx_std_17)

    # Wire the CLI to the engine + VAD.
    target_link_libraries(asr-cli PRIVATE asr_engine asr_vad)
    target_compile_definitions(asr-cli PRIVATE ASR_WITH_ENGINE)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(ASR_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    include(GoogleTest)

    # Pure-logic tests link only against asr_core (no model required).
    set(ASR_CORE_TESTS
        test_smoke
        test_parse_qwen3a
        test_chunker
        test_output_format
        test_argparse
        test_profile_registry
        test_merge
        test_carry
        test_srt_vtt
        test_postprocess
        test_dedup
    )
    foreach(t ${ASR_CORE_TESTS})
        add_executable(${t} tests/${t}.cpp)
        target_link_libraries(${t} PRIVATE asr_core GTest::gtest_main)
        gtest_discover_tests(${t})
    endforeach()

    # Engine tests build whenever the engine is built; the model-dependent ones
    # skip at runtime unless ASR_RUN_MODEL_TESTS is set (see the test bodies).
    if(ASR_BUILD_ENGINE)
        set(ASR_ENGINE_TESTS
            test_gguf_projector_type
            test_e2e_transcribe
        )
        foreach(t ${ASR_ENGINE_TESTS})
            add_executable(${t} tests/${t}.cpp)
            target_link_libraries(${t} PRIVATE asr_engine GTest::gtest_main)
            target_compile_definitions(${t} PRIVATE ASR_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
            gtest_discover_tests(${t})
        endforeach()

        # FireRedVAD tests (pure-logic ones run always; model ones gated).
        add_executable(test_vad tests/test_vad.cpp)
        target_link_libraries(test_vad PRIVATE asr_vad GTest::gtest_main)
        target_compile_definitions(test_vad PRIVATE ASR_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        gtest_discover_tests(test_vad)
    endif()
endif()
