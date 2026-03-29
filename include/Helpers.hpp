#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace mg {
std::vector<std::string> parse_file(const std::filesystem::path &filename);
std::unordered_set<char> get_uchars(std::vector<std::string> docs);
} // namespace mg