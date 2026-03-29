#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

namespace mg {
class Value : public std::enable_shared_from_this<Value> {
public:
  Value(double data, std::vector<std::shared_ptr<Value>> children = {},
        std::vector<double> local_grads = {});

  void backward();

  std::shared_ptr<Value> pow(double value);
  std::shared_ptr<Value> log();
  std::shared_ptr<Value> exp();
  std::shared_ptr<Value> relu();

  std::shared_ptr<Value> neg();

private:
  double data;
  double grad;

  // maybe could be unique_ptr.. children cannot be shared right?
  std::vector<std::shared_ptr<Value>> children;
  std::vector<double> local_grads;

  std::shared_ptr<Value> add(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> add(double value);
  std::shared_ptr<Value> sub(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> mul(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> mul(double value);
  std::shared_ptr<Value> div(const std::shared_ptr<Value> &other);

  void build_topo(const std::shared_ptr<Value> &v,
                  std::vector<std::shared_ptr<Value>> &topo,
                  std::unordered_set<std::shared_ptr<Value>> &visited);
};
} // namespace mg