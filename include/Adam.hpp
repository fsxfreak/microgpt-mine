#pragma once

#include "Model.hpp"
#include "Types.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace lmg {
class Adam {
public:
  Adam(const std::vector<std::string> &docs,
       const std::unordered_map<char, Token> &uchars, const Token bos,
       const unsigned int num_steps = DEFAULT_NUM_STEPS);

  // todo - might be better to flip the deps here and have model take
  // an optimizer for training
  void train(Model &model);

  constexpr static double LEARNING_RATE = 0.01;
  constexpr static double BETA_1 = 0.85;
  constexpr static double BETA_2 = 0.99;
  constexpr static double EPS_ADAM = 1e-8;
  constexpr static unsigned int DEFAULT_NUM_STEPS = 1000;

private:
  std::vector<unsigned int> tokenize(const std::string &doc) const;

  const std::vector<std::string> docs;
  const std::unordered_map<char, Token> uchars;
  const Token bos;
  const unsigned int num_steps;

  // first and second moment buffers
  std::vector<double> m;
  std::vector<double> v;
};
} // namespace lmg