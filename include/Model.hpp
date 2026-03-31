#pragma once

#include "Value.hpp"
#include <vector>

namespace lmg {

using Vector = std::vector<std::shared_ptr<Value>>;
using Matrix = std::vector<Vector>;
using ParamView = std::vector<Value *>;

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

  inline const ParamView &get_parameters() const { return params; }

  static Vector linear(const Vector &x, const Matrix &w);
  static Vector softmax(const Vector &logits);
  static Vector rmsnorm(const Vector &x);

private:
  Matrix wte;
  Matrix wpe;
  Matrix lm_head;

  std::vector<Layer> layers;
  ParamView params;

  // depth of transformer
  constexpr static unsigned int N_LAYER = 1;
  // width of transformer
  constexpr static unsigned int N_EMBD = 16;
  // context length of attention window
  constexpr static unsigned int BLOCK_SIZE = 16;
  // attention heads
  constexpr static unsigned int N_HEAD = 4;
  constexpr static unsigned int HEAD_DIM = N_EMBD / N_HEAD;
};
} // namespace lmg