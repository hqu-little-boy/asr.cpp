if(NOT ASR_BUILD_TESTS)
    return()
endif()

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

function(asr_add_gtest target link_target)
    add_executable(${target} ${PROJECT_SOURCE_DIR}/tests/${target}.cpp)
    target_link_libraries(${target} PRIVATE ${link_target} GTest::gtest_main)
    if(ARGN)
        target_compile_definitions(${target} PRIVATE ${ARGN})
    endif()
    gtest_discover_tests(${target})
endfunction()

# Pure-logic tests link only against asr_core (no model required).
set(ASR_CORE_TESTS
    test_smoke
    test_parse_qwen3a
    test_chunker
    test_output_format
    test_output_writer
    test_job
    test_argparse
    test_profile_registry
    test_config_error
    test_merge
    test_carry
    test_srt_vtt
    test_postprocess
    test_dedup
)
foreach(t ${ASR_CORE_TESTS})
    asr_add_gtest(${t} asr_core)
endforeach()

# Engine tests build whenever the engine is built; the model-dependent ones
# skip at runtime unless ASR_RUN_MODEL_TESTS is set (see the test bodies).
if(ASR_BUILD_ENGINE)
    set(ASR_ENGINE_TESTS
        test_gguf_projector_type
        test_e2e_transcribe
    )
    foreach(t ${ASR_ENGINE_TESTS})
        asr_add_gtest(${t} asr_engine ASR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    endforeach()

    # FireRedVAD tests (pure-logic ones run always; model ones gated).
    asr_add_gtest(test_vad asr_vad ASR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()
