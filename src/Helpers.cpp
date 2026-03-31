#include "Helpers.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <unordered_set>

namespace lmg {

std::vector<std::string> parse_file(const std::filesystem::path &filename) {
  std::ifstream infile(filename, std::ios::in);
  std::string line;
  std::vector<std::string> docs;
  while (std::getline(infile, line)) {
    docs.push_back(line);
  }

  std::mt19937_64 gen(1);
  std::shuffle(std::begin(docs), std::end(docs), gen);

  return docs;
}

std::unordered_set<char> get_uchars(std::vector<std::string> docs) {
  std::unordered_set<char> uc;
  for (auto it = docs.begin(); it != docs.end(); ++it) {
    auto &s = *it;
    for (auto sit = s.begin(); sit != s.end(); ++sit) {
      uc.emplace(*sit);
    }
  }
  return uc;
}

} // namespace lmg
