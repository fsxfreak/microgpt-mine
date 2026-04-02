#pragma once

#include "Types.hpp"
#include <unordered_map>
#include <vector>

namespace lmg {

struct Layer {
  Matrix attn_wq;
  Matrix attn_wk;
  Matrix attn_wv;
  Matrix attn_wo;
  Matrix mlp_fc1;
  Matrix mlp_fc2;

  Layer(unsigned int n_embd);
  ParamView parameters() const;
};

class Model {
public:
  Model(size_t vocab_size);

  inline ParamView &get_parameters() { return params; }

  Vector gpt(const unsigned int token_id, const unsigned int pos_id,
             std::vector<Matrix> &keys, std::vector<Matrix> &values) const;

  void inference(const Token bos, const std::unordered_map<char, Token> &uchars,
                 const double temperature = 0.5,
                 const size_t num_samples = 20) const;

  // depth of transformer
  constexpr static unsigned int N_LAYER = 1;
  // width of transformer
  constexpr static unsigned int N_EMBD = 16;
  // context length of attention window
  constexpr static unsigned int BLOCK_SIZE = 16;
  // attention heads
  constexpr static unsigned int N_HEAD = 4;
  constexpr static unsigned int HEAD_DIM = N_EMBD / N_HEAD;

private:
  Matrix wte;
  Matrix wpe;
  Matrix lm_head;

  std::vector<Layer> layers;
  ParamView params;
};
} // namespace lmg