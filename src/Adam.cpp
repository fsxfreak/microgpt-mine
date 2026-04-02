#include "Adam.hpp"

#include "Helpers.hpp"
#include <cmath>
#include <fmt/base.h>
#include <numeric>

namespace lmg {

Adam::Adam(const std::vector<std::string> &docs,
           const std::unordered_map<char, Token> &uchars, const Token bos,
           const unsigned int num_steps)
    : docs(docs), uchars(uchars), bos(bos), num_steps(num_steps) {}

std::vector<Token> Adam::tokenize(const std::string &doc) const {
  std::vector<Token> tokens{bos};
  tokens.reserve(doc.size() + 2);
  for (const auto &c : doc) {
    tokens.push_back(uchars.at(c));
  }
  tokens.push_back(bos);
  return tokens;
}

void Adam::train(Model &model) {
  auto params = model.get_parameters();
  m = std::vector<double>(params.size());
  v = std::vector<double>(params.size());
  for (size_t step = 0; step < num_steps; ++step) {
    auto doc = docs[step % docs.size()];
    auto tokens = tokenize(doc);
    unsigned int n = std::min(Model::BLOCK_SIZE,
                              static_cast<unsigned int>(tokens.size() - 1));

    // forward pass
    std::vector<Matrix> keys(Model::N_LAYER);
    std::vector<Matrix> values(Model::N_LAYER);
    std::vector<std::shared_ptr<Value>> losses;
    losses.reserve(n);
    for (size_t pos_id = 0; pos_id < n; ++pos_id) {
      auto token_id = tokens[pos_id];
      auto target_id = tokens[pos_id + 1];
      auto logits = model.gpt(token_id, pos_id, keys, values);
      auto probs = softmax(logits);
      auto loss_t = probs[target_id]->log()->neg();
      losses.push_back(loss_t);
    }
    auto loss =
        (1.0 / n) * std::accumulate(losses.begin(), losses.end(),
                                    std::make_shared<Value>(0.0),
                                    [](const auto &accum, const auto &other) {
                                      return accum + other;
                                    });

    loss->backward();

    double lr_t = LEARNING_RATE * (1 - static_cast<double>(step) / num_steps);
    for (size_t i = 0; i < params.size(); ++i) {
      m[i] = BETA_1 * m[i] + (1 - BETA_1) * params[i]->get_grad();
      v[i] = BETA_2 * v[i] + (1 - BETA_2) * std::pow(params[i]->get_grad(), 2);
      auto m_hat = m[i] / (1 - std::pow(BETA_1, step + 1));
      auto v_hat = v[i] / (1 - std::pow(BETA_2, step + 1));
      double update_delta = (lr_t * m_hat) / (std::pow(v_hat, 0.5) + EPS_ADAM);
      params[i]->update(-update_delta);
    }
    if (step % 50 == 0) {
      fmt::println("step {:4d} / {:4d} | loss {:.4f}", step, num_steps,
                   loss->get_data());
    }
  }
}
} // namespace lmg