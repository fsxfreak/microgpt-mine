#pragma once

#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace lmg {
std::vector<std::string> parse_file(const std::filesystem::path &filename);
std::unordered_set<char> get_uchars(std::vector<std::string> docs);
} // namespace lmg