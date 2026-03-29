#include "Value.hpp"

#include <cassert>
#include <fmt/base.h>
#include <ranges>
#include <unordered_set>

namespace mg {
Value::Value(double data, std::vector<std::shared_ptr<Value>> children,
             std::vector<double> local_grads)
    : data(data), grad(0), children(std::move(children)),
      local_grads(std::move(local_grads)) {}

void Value::backward() {
  std::vector<std::shared_ptr<Value>> topo;
  std::unordered_set<std::shared_ptr<Value>, ValuePtrHash> visited;
  build_topo(shared_from_this(), topo, visited);

  grad = 1.0;

  for (const auto &value : std::views::reverse(topo)) {
    const std::vector<std::shared_ptr<Value>> &c = value->children;
    const std::vector<double> &lg = value->local_grads;
    assert(c.size() == lg.size());

    for (size_t i = 0; i < c.size(); ++i) {
      c.at(i)->grad += lg.at(i) * value->grad;
    }
  }
}

void Value::build_topo(
    const std::shared_ptr<Value> &v, std::vector<std::shared_ptr<Value>> &topo,
    std::unordered_set<std::shared_ptr<Value>, ValuePtrHash> &visited) {
  if (visited.contains(v)) {
    return;
  }
  visited.insert(v);
  for (const auto &child : v->children) {
    build_topo(child, topo, visited);
  }
  topo.push_back(v);
}

std::shared_ptr<Value> Value::add(const std::shared_ptr<Value> &other) {
  return std::make_shared<Value>(
      data + other->data,
      std::vector<std::shared_ptr<Value>>{shared_from_this(), other},
      std::vector<double>{1.0, 1.0});
}
std::shared_ptr<Value> Value::add(double value) {
  return add(std::make_shared<Value>(value));
}
std::shared_ptr<Value> Value::sub(const std::shared_ptr<Value> &other) {
  return add(other->neg());
}
std::shared_ptr<Value> Value::mul(const std::shared_ptr<Value> &other) {
  return std::make_shared<Value>(
      data * other->data,
      std::vector<std::shared_ptr<Value>>{shared_from_this(), other},
      std::vector<double>{other->data, data});
}
std::shared_ptr<Value> Value::mul(double value) {
  return mul(std::make_shared<Value>(value));
}
std::shared_ptr<Value> Value::div(const std::shared_ptr<Value> &other) {
  return mul(other->pow(-1.0));
}

std::shared_ptr<Value> Value::pow(double value) {
  return std::make_shared<Value>(
      std::pow(data, value),
      std::vector<std::shared_ptr<Value>>{shared_from_this()},
      std::vector<double>{value * std::pow(data, value - 1)});
}
std::shared_ptr<Value> Value::log() {
  return std::make_shared<Value>(
      std::log(data), std::vector<std::shared_ptr<Value>>{shared_from_this()},
      std::vector<double>{1.0 / data});
}
std::shared_ptr<Value> Value::exp() {
  return std::make_shared<Value>(
      std::exp(data), std::vector<std::shared_ptr<Value>>{shared_from_this()},
      std::vector<double>{std::exp(data)});
}
std::shared_ptr<Value> Value::relu() {
  return std::make_shared<Value>(
      std::max(0.0, data),
      std::vector<std::shared_ptr<Value>>{shared_from_this()},
      std::vector<double>{data > 0.0 ? 1.0 : 0.0});
}

std::shared_ptr<Value> Value::neg() { return mul(-1.0); }

void Value::debug_print(int tabs) const {
  std::string tab_str(tabs, '\t');

  fmt::print("{}d:{}, g:{}\n", tab_str, data, grad);
  for (const auto &c : children) {
    c->debug_print(tabs + 1);
  }
}
} // namespace mg