#include "Helpers.hpp"
#include "Types.hpp"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <ranges>
#include <set>
#include <unordered_map>

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

std::unordered_map<char, Token> get_uchars(std::vector<std::string> docs) {
  std::set<char> uc;
  for (auto it = docs.begin(); it != docs.end(); ++it) {
    auto &s = *it;
    for (auto sit = s.begin(); sit != s.end(); ++sit) {
      uc.emplace(*sit);
    }
  }

  std::unordered_map<char, Token> uc_mapped;
  size_t i = 0;
  for (const auto &c : uc) {
    // std set should order chars alphabetically - technically this doesn't
    // matter, we just need a token assignment
    uc_mapped[c] = i++;
  }
  return uc_mapped;
}

Vector linear(const Vector &x, const Matrix &w) {
  Vector result;
  for (const auto &wo : w) {
    assert(wo.size() == x.size());
    auto accum = std::make_shared<Value>(0);
    for (const auto [wi, xi] : std::views::zip(wo, x)) {
      accum = accum + (wi * xi);
    }
    result.push_back(accum);
  }
  return result;
}

Vector softmax(const Vector &logits) {
  auto max_value_it = std::max_element(
      logits.begin(), logits.end(),
      [](const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs) {
        return lhs->get_data() < rhs->get_data();
      });
  Vector exps;
  exps.reserve(logits.size());
  std::transform(logits.cbegin(), logits.cend(), std::back_inserter(exps),
                 [max_value_it](const std::shared_ptr<Value> &v) {
                   return (v - *max_value_it)->exp();
                 });
  assert(exps.size() == logits.size());

  auto total =
      std::accumulate(exps.cbegin(), exps.cend(), std::make_shared<Value>(0.0),
                      [](std::shared_ptr<Value> accum,
                         std::shared_ptr<Value> next) { return accum + next; });
  std::transform(
      exps.cbegin(), exps.cend(), exps.begin(),
      [total](const std::shared_ptr<Value> &e) { return e / total; });
  return exps;
}

Vector rmsnorm(const Vector &x) {
  double mean_square =
      std::accumulate(x.cbegin(), x.cend(), 0.0,
                      [](double accum, const std::shared_ptr<Value> &next) {
                        return accum + (next->get_data() * next->get_data());
                      }) /
      static_cast<double>(x.size());
  double scale = std::pow(mean_square + 1e-5, -0.5);
  Vector result;
  result.reserve(x.size());
  std::transform(
      x.cbegin(), x.cend(), std::back_inserter(result),
      [scale](const std::shared_ptr<Value> &x) { return x * scale; });
  return result;
}

} // namespace lmg
