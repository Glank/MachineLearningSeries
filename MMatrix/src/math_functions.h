#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

namespace math {

inline double sigmoid(double x) {
  return 1.0/(1.0+std::exp(-x));
}

inline double sigmoid_deriv(double x) {
  double s = sigmoid(x);
  return s*(1.0-s);
}

}  // namespace math

#endif // MATH_FUNCTIONS_H
