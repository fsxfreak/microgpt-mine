#pragma once

#include <memory>
#include <vector>

#include "Value.hpp"

namespace lmg {
using Vector = std::vector<std::shared_ptr<Value>>;
using Matrix = std::vector<Vector>;
using ParamView = std::vector<Value *>;
} // namespace lmg
