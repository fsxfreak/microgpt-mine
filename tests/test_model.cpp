#include "Model.hpp"
#include "Types.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Model tests", "[math]") {
  constexpr unsigned int VOCAB_SIZE = 8;
  auto m = lmg::Model(VOCAB_SIZE);
  std::vector<lmg::Matrix> keys(lmg::Model::N_LAYER);
  std::vector<lmg::Matrix> values(lmg::Model::N_LAYER);

  SECTION("Gpt calculate probability distribution on vocab size") {
    auto res = m.gpt(0, 0, keys, values);
    REQUIRE(res.size() == VOCAB_SIZE);
  }
}