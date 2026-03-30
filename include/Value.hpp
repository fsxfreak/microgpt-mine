#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

namespace lmg {

struct ValuePtrHash;

class Value : public std::enable_shared_from_this<Value> {
public:
  Value(double data, std::vector<std::shared_ptr<Value>> children = {},
        std::vector<double> local_grads = {});

  void backward();

  std::shared_ptr<Value> add(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> add(double value);
  std::shared_ptr<Value> sub(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> mul(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> mul(double value);
  std::shared_ptr<Value> div(const std::shared_ptr<Value> &other);

  std::shared_ptr<Value> pow(double value);
  std::shared_ptr<Value> log();
  std::shared_ptr<Value> exp();
  std::shared_ptr<Value> relu();

  std::shared_ptr<Value> neg();

  inline double getData() const { return data; }
  inline double getGrad() const { return grad; }
  inline const std::vector<std::shared_ptr<Value>> &get_children() const {
    return children;
  }

  void debug_print(int tabs = 0) const;

private:
  double data;
  double grad;

  std::vector<std::shared_ptr<Value>> children;
  std::vector<double> local_grads;

  void
  build_topo(const std::shared_ptr<Value> &v,
             std::vector<std::shared_ptr<Value>> &topo,
             std::unordered_set<std::shared_ptr<Value>, ValuePtrHash> &visited);
};

inline std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs,
                                        const std::shared_ptr<Value> &rhs) {
  return lhs->add(rhs);
}
inline std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs,
                                        double rhs) {
  return lhs->add(rhs);
}
inline std::shared_ptr<Value> operator+(double lhs,
                                        const std::shared_ptr<Value> &rhs) {
  return rhs->add(lhs);
}
inline std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &lhs,
                                        const std::shared_ptr<Value> &rhs) {
  return lhs->mul(rhs);
}
inline std::shared_ptr<Value> operator*(double lhs,
                                        const std::shared_ptr<Value> &rhs) {
  return rhs->mul(lhs);
}
inline std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &lhs,
                                        const double rhs) {
  return rhs * lhs;
}
inline std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &lhs,
                                        const double rhs) {
  return lhs + -rhs;
}
inline std::shared_ptr<Value> operator-(double lhs,
                                        const std::shared_ptr<Value> &rhs) {
  return lhs + rhs->neg();
}
inline std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &lhs,
                                        const std::shared_ptr<Value> &rhs) {
  return lhs * rhs->pow(-1.0);
}
inline std::shared_ptr<Value> operator/(double lhs,
                                        const std::shared_ptr<Value> &rhs) {
  return lhs * rhs->pow(-1.0);
}
inline std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &lhs,
                                        const double rhs) {
  return lhs * (1.0 / rhs);
}

struct ValuePtrHash {
  std::size_t operator()(const std::shared_ptr<Value> &ptr) const {
    return reinterpret_cast<std::size_t>(ptr.get());
  }
};

} // namespace lmg