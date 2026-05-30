#include "asr_output.h"

#include <cstdio>

namespace asr {

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) c);
                    out += buf;
                } else {
                    out += (char) c; // printable ASCII or UTF-8 continuation byte
                }
        }
    }
    return out;
}

void write_txt(std::ostream & os, const result & r) {
    os << r.text << "\n";
}

void write_json(std::ostream & os, const result & r) {
    os << "{\n";
    os << "  \"language\": \"" << json_escape(r.language) << "\",\n";
    os << "  \"text\": \""     << json_escape(r.text)     << "\"\n";
    os << "}\n";
}

} // namespace asr
