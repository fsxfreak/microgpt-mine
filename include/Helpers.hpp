#pragma once

#include <filesystem>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "Types.hpp"

namespace lmg {
std::vector<std::string> parse_file(const std::filesystem::path &filename);
std::unordered_map<char, Token> get_uchars(std::vector<std::string> docs);

Vector linear(const Vector &x, const Matrix &w);
Vector softmax(const Vector &logits);
Vector rmsnorm(const Vector &x);
} // namespace lmg