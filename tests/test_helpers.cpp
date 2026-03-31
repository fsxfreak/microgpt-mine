
#include "Helpers.hpp"
#include "Value.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numeric>

TEST_CASE("Matrix/vector math works correctly", "[math]") {
  auto x = lmg::Vector{std::make_shared<lmg::Value>(5.0),
                       std::make_shared<lmg::Value>(6.0)};
  auto w = lmg::Matrix{lmg::Vector{std::make_shared<lmg::Value>(1.0),
                                   std::make_shared<lmg::Value>(2.0)},
                       lmg::Vector{std::make_shared<lmg::Value>(3.0),
                                   std::make_shared<lmg::Value>(4.0)}};

  SECTION("Matrix vector multiply") {
    auto res = lmg::linear(x, w);
    REQUIRE(res[0]->get_data() == 17.0);
    REQUIRE(res[1]->get_data() == 39.0);
  }

  SECTION("Softmax") {
    auto res = lmg::softmax(x);
    REQUIRE(res.size() == x.size());
    auto accum = std::accumulate(
        res.begin(), res.end(), std::make_shared<lmg::Value>(0.0),
        [](std::shared_ptr<lmg::Value> accum,
           std::shared_ptr<lmg::Value> next) { return accum + next; });
    REQUIRE(accum->get_data() == 1.0);
  }

  SECTION("rmsnorm") {
    auto x = lmg::Vector{std::make_shared<lmg::Value>(2.0),
                         std::make_shared<lmg::Value>(3.0)};
    auto res = lmg::rmsnorm(x);
    REQUIRE(res.size() == x.size());
    REQUIRE_THAT(res[0]->get_data(),
                 Catch::Matchers::WithinRel(0.78446152, 0.01));
    REQUIRE_THAT(res[1]->get_data(),
                 Catch::Matchers::WithinRel(1.17669229, 0.01));
  }
}