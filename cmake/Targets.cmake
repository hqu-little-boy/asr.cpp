# ---------------------------------------------------------------------------
# asr_core: pure-logic library (NO llama.cpp / mtmd dependency).
# Holds parsing, chunking, formatting, argument parsing, profile registry and
# merging, so the bulk of the logic is unit-testable without a model.
# ---------------------------------------------------------------------------
add_library(asr_core STATIC
    ${PROJECT_SOURCE_DIR}/src/asr_types.cpp
    ${PROJECT_SOURCE_DIR}/src/asr_profile.cpp
    ${PROJECT_SOURCE_DIR}/src/asr_chunker.cpp
    ${PROJECT_SOURCE_DIR}/src/asr_output.cpp
    ${PROJECT_SOURCE_DIR}/src/asr_args.cpp
    ${PROJECT_SOURCE_DIR}/src/asr_merge.cpp
    ${PROJECT_SOURCE_DIR}/src/asr_carry.cpp
    ${PROJECT_SOURCE_DIR}/src/asr_postprocess.cpp
)
target_include_directories(asr_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(asr_core PUBLIC cxx_std_17)
target_link_libraries(asr_core PRIVATE CLI11::CLI11)

# ---------------------------------------------------------------------------
# asr-cli: command-line front-end (thin shell over the library).
# ---------------------------------------------------------------------------
add_executable(asr-cli ${PROJECT_SOURCE_DIR}/cli/asr_cli.cpp)
target_link_libraries(asr-cli PRIVATE asr_core)

# ---------------------------------------------------------------------------
# asr_engine: mtmd-backed engine (optional).
# ---------------------------------------------------------------------------
if(ASR_BUILD_ENGINE)
    # FireRedVAD (DFSMN) voice-activity detector; links only ggml (for gguf).
    add_library(asr_vad STATIC ${PROJECT_SOURCE_DIR}/src/asr_vad.cpp)
    target_include_directories(asr_vad PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(asr_vad PUBLIC ggml)
    target_compile_features(asr_vad PRIVATE cxx_std_17)

    # mtmd-backed engine library.
    add_library(asr_engine STATIC
        ${PROJECT_SOURCE_DIR}/src/asr_gguf.cpp
        ${PROJECT_SOURCE_DIR}/src/asr_audio.cpp
        ${PROJECT_SOURCE_DIR}/src/asr_engine.cpp
        ${PROJECT_SOURCE_DIR}/src/asr_driver.cpp
    )
    target_include_directories(asr_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(asr_engine PUBLIC asr_core mtmd llama-common)
    target_compile_features(asr_engine PRIVATE cxx_std_17)

    # Wire the CLI to the engine + VAD.
    target_link_libraries(asr-cli PRIVATE asr_engine asr_vad)
    target_compile_definitions(asr-cli PRIVATE ASR_WITH_ENGINE)
endif()
