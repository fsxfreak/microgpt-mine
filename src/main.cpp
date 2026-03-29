#include <iostream>
#include <string_view>

#include "Helpers.hpp"
#include "Value.hpp"

constexpr std::string_view FILE_INPUT = "../data/names.txt";

int main() {
  auto docs = mg::parse_file(FILE_INPUT);

  std::cout << "have " << docs.size() << " docs" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << docs.at(i) << std::endl;
  }

  auto uchars = mg::get_uchars(docs);
  // int BOS = uchars.size();
  size_t vocab_size = uchars.size() + 1;
  std::cout << "have " << vocab_size << " vocab size" << std::endl;

  auto a = std::make_shared<mg::Value>(2.0);
  auto b = std::make_shared<mg::Value>(3.0);
  auto c = a + b;
  auto d = 7.0 * c;
  d->backward();
  d->debug_print();
}