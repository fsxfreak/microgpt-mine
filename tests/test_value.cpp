#include "Value.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Value math works correctly", "[math]") {
  auto a = std::make_shared<lmg::Value>(2.0);
  auto b = std::make_shared<lmg::Value>(3.0);

  SECTION("Value + Value") {
    auto c = a + b;
    REQUIRE(c->getData() == 5.0);
    // Verify graph was built
    REQUIRE(c->getChildren().size() == 2);
  }

  SECTION("Value + double") {
    auto c = a + 10.0;
    REQUIRE(c->getData() == 12.0);
  }

  SECTION("double + Value (Reverse Operand)") {
    auto c = 10.0 + a;
    REQUIRE(c->getData() == 12.0);
  }

  SECTION("Div works") {
    auto c = (a + b) / (b + a) + 1.0;
    REQUIRE(c->getData() == 2.0);
  }

  SECTION("Backward") {
    auto c = 5.0 * a;
    c->backward();
    REQUIRE(a->getGrad() == 5.0);
  }
}