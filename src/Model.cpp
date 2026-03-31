#include "Model.hpp"

#include <algorithm>
#include <fmt/base.h>
#include <random>

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

} // namespace lmg
