cmake_minimum_required(VERSION 3.18)
project(asr_cpp LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ASR_BUILD_ENGINE "Build the mtmd-backed engine and full CLI (requires llama.cpp)" OFF)
option(ASR_BUILD_TESTS  "Build unit tests (GoogleTest)" ON)

include(cmake/Dependencies.cmake)
asr_configure_dependencies()

include(cmake/Targets.cmake)
include(cmake/Testing.cmake)
