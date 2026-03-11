#include <fstream>
#include <string>

constexpr std::string FILE_INPUT = "data/names.txt";

int main() { std::ifstream infile(FILE_INPUT, "r"); }