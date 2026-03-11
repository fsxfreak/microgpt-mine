#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>

constexpr std::string FILE_INPUT = "../data/names.txt";

std::vector<std::string> parse_file() {
  std::ifstream infile(FILE_INPUT, std::ios::in);
  std::string line;
  std::vector<std::string> docs;
  while (std::getline(infile, line)) {
    docs.push_back(line);
  }

  std::mt19937_64 gen(1);
  std::shuffle(std::begin(docs), std::end(docs), gen);

  return docs;
}

decltype(auto) get_uchars(std::vector<std::string> docs) {
  std::set<char> uc;
  for (auto it = docs.begin(); it != docs.end(); ++it) {
    auto &s = *it;
    for (auto sit = s.begin(); sit != s.end(); ++sit) {
      uc.emplace(*sit);
    }
  }
  return uc;
}

class Value {
public:
  Value() {}
};

int main() {
  auto docs = parse_file();

  std::cout << "have " << docs.size() << " docs" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << docs.at(i) << std::endl;
  }

  auto uchars = get_uchars(docs);
  // int BOS = uchars.size();
  size_t vocab_size = uchars.size() + 1;
  std::cout << "have " << vocab_size << " vocab size" << std::endl;
}