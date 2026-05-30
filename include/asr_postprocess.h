#pragma once

#include "asr.h"

#include <string>

namespace asr {

// Collapse runs of a single codepoint repeated more than `threshold` times
// into at most `threshold` repetitions. E.g. "哈哈哈哈…" (30×) → "哈哈" (2×).
// Handles multi-byte UTF-8 correctly (counts by codepoints, not bytes).
std::string fix_char_repeats(const std::string & s, int threshold = 20);

// Collapse an n-gram repeated `threshold` or more consecutive times.
// Uses n=2..6 codepoints, checks for ≥threshold consecutive repetitions.
// E.g. "你好世界你好世界你好世界…" (20×) → "你好世界" (1×).
std::string fix_pattern_repeats(const std::string & s, int threshold = 20);

// Apply both fix_char_repeats and fix_pattern_repeats.
std::string suppress_repeats(const std::string & s, int threshold = 20);

// Remove inter-segment duplicates: if two adjacent segments have text where
// one is a prefix of the other (or they share a very long common prefix),
// the duplicate is dropped. Modifies `r.segments` and rebuilds `r.text`.
// `min_common` is the minimum number of codepoints for a prefix to be
// considered a duplicate (default 8).
void dedup_segments(result & r, size_t min_common = 8);

} // namespace asr
