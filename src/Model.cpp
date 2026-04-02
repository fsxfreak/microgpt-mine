#include "Model.hpp"
#include "Helpers.hpp"

#include <algorithm>
#include <cassert>
#include <fmt/base.h>
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
    : wte(initialize_matrix(vocab_size, N_EMBD)),
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

Vector Model::gpt(const unsigned int token_id, const unsigned int pos_id,
                  std::vector<Matrix> &keys,
                  std::vector<Matrix> &values) const {
  assert(keys.size() == N_LAYER);
  assert(values.size() == N_LAYER);
  const auto &tok_emb = wte.at(token_id);
  const auto &pos_emb = wpe.at(pos_id);
  Vector x;
  x.reserve(tok_emb.size());
  for (const auto [t, p] : std::views::zip(tok_emb, pos_emb)) {
    x.emplace_back(t + p);
  }
  x = rmsnorm(x);

  for (size_t li = 0; li < N_LAYER; ++li) {
    // Multi head attention block
    Vector x_residual(x);
    x = rmsnorm(x);
    Vector q = linear(x, layers.at(li).attn_wq);
    Vector k = linear(x, layers.at(li).attn_wk);
    Vector v = linear(x, layers.at(li).attn_wv);

    keys.at(li).push_back(k);
    values.at(li).push_back(v);
    Vector x_attn;
    x_attn.reserve(N_HEAD * HEAD_DIM);
    for (size_t h = 0; h < N_HEAD; ++h) {
      size_t hs = h * HEAD_DIM;

      auto q_h = std::span(q).subspan(hs, HEAD_DIM);
      auto k_h = keys.at(li) | std::views::transform([hs](const Vector &ki) {
                   return std::span(ki).subspan(hs, HEAD_DIM);
                 });
      auto v_h = values.at(li) | std::views::transform([hs](const Vector &vi) {
                   return std::span(vi).subspan(hs, HEAD_DIM);
                 });

      Vector attn_logits;
      attn_logits.reserve(k_h.size());
      for (size_t t = 0; t < k_h.size(); ++t) {
        // Lazily evaluate k_h[t] once
        auto k_h_t = k_h[t];
        // Initialize accumulator with the first value to avoid creating a
        // starting accuulator
        auto accum = q_h[0] + k_h_t[0];
        for (size_t j = 1; j < HEAD_DIM; ++j) {
          accum = accum + (q_h[j] * k_h_t[j]);
        }
        accum = accum / std::pow(HEAD_DIM, 0.5);
        attn_logits.push_back(accum);
      }
      auto attn_weights = softmax(attn_logits);

      Vector head_out;
      head_out.reserve(HEAD_DIM);
      // Invert the calculation here to fit locality better
      auto v_h_0 = v_h[0];
      auto w_0 = attn_weights.at(0);
      for (size_t j = 0; j < HEAD_DIM; ++j) {
        head_out.push_back(w_0 * v_h_0[j]);
      }
      for (size_t t = 1; t < v_h.size(); ++t) {
        auto v_h_t = v_h[t];
        auto w_t = attn_weights.at(t);
        for (size_t j = 0; j < HEAD_DIM; ++j) {
          head_out[j] = head_out[j] + (w_t * v_h_t[j]);
        }
      }
      x_attn.insert(x_attn.end(), head_out.begin(), head_out.end());
    }

    x = linear(x_attn, layers.at(li).attn_wo);
    for (auto &&[a, b] : std::views::zip(x, x_residual)) {
      a = a + b;
    }

    // MLP block
    x_residual = x;
    x = rmsnorm(x);
    x = linear(x, layers.at(li).mlp_fc1);

    std::transform(x.cbegin(), x.cend(), x.begin(),
                   [](const std::shared_ptr<Value> &xi) { return xi->relu(); });
    x = linear(x, layers.at(li).mlp_fc2);
    for (auto &&[a, b] : std::views::zip(x, x_residual)) {
      a = a + b;
    }
  }

  auto logits = linear(x, lm_head);
  return logits;
}

std::unordered_map<Token, char>
invert_char_map(const std::unordered_map<char, Token> &uchars) {
  std::unordered_map<Token, char> result;
  for (const auto [c, t] : uchars) {
    assert(!result.contains(t));
    result[t] = c;
  }
  return result;
}

Token weighted_sample(const Vector &probs, std::mt19937 &gen) {
  std::vector<double> probs_data;
  probs_data.reserve(probs_data.size());
  std::transform(
      probs.begin(), probs.end(), std::back_inserter(probs_data),
      [](const std::shared_ptr<Value> &prob) { return prob->get_data(); });

  std::discrete_distribution<> d(probs_data.begin(), probs_data.end());
  return d(gen);
}

void Model::inference(const Token bos,
                      const std::unordered_map<char, Token> &uchars,
                      const double temperature,
                      const size_t num_samples) const {
  fmt::println("inference temp {}", temperature);

  std::random_device rd;
  std::mt19937 gen(rd());

  auto token_map = invert_char_map(uchars);
  for (size_t i = 0; i < num_samples; ++i) {
    std::vector<Matrix> keys(N_LAYER);
    std::vector<Matrix> values(N_LAYER);
    Token token_id = bos;
    std::vector<char> sample;
    for (Token pos_id = 0; pos_id < BLOCK_SIZE; ++pos_id) {
      auto logits = gpt(token_id, pos_id, keys, values);
      std::transform(logits.cbegin(), logits.cend(), logits.begin(),
                     [temperature](const std::shared_ptr<Value> &logit) {
                       return logit / temperature;
                     });
      auto probs = softmax(logits);
      token_id = weighted_sample(probs, gen);
      if (token_id == bos) {
        break;
      }
      sample.push_back(token_map[token_id]);
    }
    std::string s{sample.begin(), sample.end()};
    fmt::println("sample {}: {}", i, s);
  }
}

} // namespace lmg
