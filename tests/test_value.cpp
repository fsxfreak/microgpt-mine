#include "Value.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Value addition works correctly", "[math]") {
  auto a = std::make_shared<mg::Value>(2.0);
  auto b = std::make_shared<mg::Value>(3.0);

  SECTION("Value + Value") {
    auto c = a + b;
    REQUIRE(c->data == 5.0);
    // Verify graph was built
    REQUIRE(c->children.size() == 2);
  }

  SECTION("Value + double") {
    auto c = a + 10.0;
    REQUIRE(c->data == 12.0);
  }

  SECTION("double + Value (Reverse Operand)") {
    auto c = 10.0 + a;
    REQUIRE(c->data == 12.0);
  }
}