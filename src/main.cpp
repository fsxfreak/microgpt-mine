#include <iostream>
#include <string_view>

#include "Helpers.hpp"

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
}