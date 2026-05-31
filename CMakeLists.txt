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

include(cmake/Dependencies.cmake)
include(cmake/Targets.cmake)
include(cmake/Testing.cmake)
