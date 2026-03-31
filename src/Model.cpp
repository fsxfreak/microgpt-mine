#include "Model.hpp"

#include <algorithm>
#include <cassert>
#include <fmt/base.h>
#include <numeric>
#include <random>
#include <ranges>

namespace lmg {
Matrix initialize_matrix(size_t nout, size_t nin, double std = 0.8) {
  std::mt19937_64 engine(1);
  std::normal_distribution d{0.0, std};

  auto gen = [&]() { return std::make_shared<Value>(d(engine)); };
  Matrix matrix(nout);
  for (size_t i = 0; i < nout; ++i) {
    std::vector<std::shared_ptr<Value>> row;
    row.reserve(nin);
    std::generate_n(std::back_inserter(row), nin, gen);
    matrix[i] = row;
  }
  return matrix;
}

ParamView flatten(const Matrix &m) {
  ParamView params;
  size_t total_size = 0;
  for (const auto &row : m) {
    total_size += row.size();
  }
  params.reserve(total_size);
  for (const auto &row : m) {
    for (const auto &val : row) {
      params.push_back(val.get());
    }
  }
  return params;
}

Layer::Layer(unsigned int n_embd)
    : attn_wq(initialize_matrix(n_embd, n_embd)),
      attn_wk(initialize_matrix(n_embd, n_embd)),
      attn_wv(initialize_matrix(n_embd, n_embd)),
      attn_wo(initialize_matrix(n_embd, n_embd)),
      mlp_fc1(initialize_matrix(4 * n_embd, n_embd)),
      mlp_fc2(initialize_matrix(n_embd, 4 * n_embd)) {}

ParamView Layer::parameters() const {
  ParamView params;

  for (const auto &matrix :
       {&attn_wq, &attn_wk, &attn_wv, &attn_wo, &mlp_fc1, &mlp_fc2}) {
    auto matrix_params = flatten(*matrix);
    params.insert(params.end(), matrix_params.begin(), matrix_params.end());
  }

  return params;
}

Model::Model(size_t vocab_size)
    : wte(initialize_matrix(N_EMBD, N_EMBD)),
      wpe(initialize_matrix(BLOCK_SIZE, N_EMBD)),
      lm_head(initialize_matrix(vocab_size, N_EMBD)) {
  layers.reserve(N_LAYER);
  for (size_t i = 0; i < N_LAYER; ++i) {
    layers.emplace_back(Layer(N_EMBD));
  }

  for (const Matrix *m : {&wte, &wpe, &lm_head}) {
    auto matrix_params = flatten(*m);
    params.insert(params.end(), matrix_params.begin(), matrix_params.end());
  }

  for (auto &l : layers) {
    auto layer_params = l.parameters();
    params.insert(params.end(), layer_params.begin(), layer_params.end());
  }
}

Vector Model::linear(const Vector &x, const Matrix &w) {
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

Vector Model::softmax(const Vector &logits) {
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

Vector Model::rmsnorm(const Vector &x) {
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
