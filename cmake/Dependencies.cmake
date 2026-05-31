function(asr_configure_dependencies)
    find_package(CLI11 REQUIRED)

    if(ASR_BUILD_ENGINE)
        # llama.cpp is a vendored dependency. Configure it only through public
        # cache options from this project; do not modify dependency sources.
        set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
        add_subdirectory(${PROJECT_SOURCE_DIR}/llama.cpp ${PROJECT_BINARY_DIR}/llama.cpp)

        # tools/mtmd is added directly (not via llama.cpp's own tools/ subdir), so it
        # does not inherit LLAMA_INSTALL_VERSION from llama.cpp's subdirectory scope.
        # Define it here so mtmd's set_target_properties(VERSION ...) is well-formed.
        set(LLAMA_INSTALL_VERSION 0.0.0)
        add_subdirectory(${PROJECT_SOURCE_DIR}/llama.cpp/tools/mtmd ${PROJECT_BINARY_DIR}/mtmd-build)
    endif()
endfunction()
