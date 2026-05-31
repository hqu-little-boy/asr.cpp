#pragma once

#include <filesystem>
#include <source_location>
#include <string>

namespace asr {

enum class error_code {
    invalid_argument,
    validation,
    io,
    model,
    vad,
    engine,
    output,
    internal,
};

struct asr_error {
    error_code           code = error_code::internal;
    std::string          stage;
    std::string          message;
    std::filesystem::path path;
    std::source_location where = std::source_location::current();
};

inline asr_error make_error(error_code code,
                            std::string message,
                            std::string stage = {},
                            std::filesystem::path path = {},
                            std::source_location where = std::source_location::current()) {
    asr_error err;
    err.code    = code;
    err.stage   = std::move(stage);
    err.message = std::move(message);
    err.path    = std::move(path);
    err.where   = where;
    return err;
}

} // namespace asr
