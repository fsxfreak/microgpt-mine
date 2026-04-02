#include <fmt/base.h>
#include <iostream>
#include <string_view>

#include "Adam.hpp"
#include "Helpers.hpp"
#include "Model.hpp"

constexpr std::string_view FILE_INPUT = "../data/names.txt";

int main() {
  auto docs = lmg::parse_file(FILE_INPUT);

  std::cout << "have " << docs.size() << " docs" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << docs.at(i) << std::endl;
  }

  auto uchars = lmg::get_uchars(docs);
  size_t vocab_size = uchars.size() + 1;
  std::cout << "have " << vocab_size << " vocab size" << std::endl;

  lmg::Adam adam(docs, uchars, uchars.size());
  lmg::Model model(vocab_size);
  adam.train(model);
  model.inference(uchars.size(), uchars);
}