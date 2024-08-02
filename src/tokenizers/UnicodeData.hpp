// MIT License

// Copyright (c) 2023-2024 The ggml authors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct range_nfd {
    uint32_t first;
    uint32_t last;
    uint32_t nfd;
};

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const std::unordered_set<uint32_t> unicode_set_whitespace;
extern const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase;
extern const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase;
extern const std::vector<range_nfd> unicode_ranges_nfd;
