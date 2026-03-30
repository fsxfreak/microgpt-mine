#include <fmt/base.h>
#include <iostream>
#include <string_view>

#include "Helpers.hpp"
#include "Model.hpp"
#include "Value.hpp"

constexpr std::string_view FILE_INPUT = "../data/names.txt";

int main() {
  auto docs = lmg::parse_file(FILE_INPUT);

  std::cout << "have " << docs.size() << " docs" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << docs.at(i) << std::endl;
  }

  auto uchars = lmg::get_uchars(docs);
  // int BOS = uchars.size();
  size_t vocab_size = uchars.size() + 1;
  std::cout << "have " << vocab_size << " vocab size" << std::endl;

  auto a = std::make_shared<lmg::Value>(2.0);
  auto b = std::make_shared<lmg::Value>(3.0);
  auto c = a + b;
  auto d = 7.0 * c;
  d->backward();
  d->debug_print();

  lmg::Model model(vocab_size);
  lmg::ParamView params = model.get_parameters();
  fmt::println("Param size: {}", params.size());
  for (size_t i = 0; i < params.size(); ++i) {
    if (i > 8) {
      break;
    }

    fmt::println("{}", params.at(i)->getData());
  }
}