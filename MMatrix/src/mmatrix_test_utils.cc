#include "mmatrix_test_utils.h"

#include <iostream>

using std::vector;

namespace mmatrix {
namespace test {

MMFloat randf() {
  return rand()/static_cast<MMFloat>(RAND_MAX);
}

vector<int> RandShape(int length, int max) {
  if (length == -1) length = rand()%3;
  vector<int> s;
  s.reserve(length);
  for (int i = 0; i < length; i++) {
    int val = rand()%max+1;
    s.push_back(val);
  }
  return s;
}

vector<int> RandIndex(const vector<int>& shape) {
  vector<int> idx;
  idx.reserve(shape.size());
  for (int i = 0; i < shape.size(); i++) {
    idx.push_back(rand()%shape[i]);
  }
  return idx;
}

void RandMatrix(MMatrixInterface* out, MMFloat zerochance, int min, int max) {
  // zerochance in [0.0,1.0]
  // 1.0 would mean always zero
  int ncells = 1;
  for (int s : out->shape()) {
    ncells *= s;
  }
  for (int i = 0; i < ncells; i++) {
    if (randf() < zerochance) continue;
    out->set(i, rand()%(max-min+1)+min);
  }
}

// Approximate df/dx(x)[f_index + x_index]
MMFloat ApproxDerivative(MMFunction f, MMatrixInterface* out_tmp,
    MMatrixInterface* x, vector<int> f_index, vector<int> x_index, MMFloat epsilon) {
  MMFloat x1 = x->get(x_index);
  f(x, out_tmp);
  MMFloat y1 = out_tmp->get(f_index);
  MMFloat x2 = x1+epsilon;
  x->set(x_index, x2);
  f(x, out_tmp);
  MMFloat y2 = out_tmp->get(f_index);
  x->set(x_index, x1);
  MMFloat ret = (y2-y1)/epsilon;
  return ret;
}

vector<int> ConcatNTimes(int n, const vector<int>& x) {
  vector<int> out;
  out.reserve(n*x.size());
  for (int i = 0; i < n*x.size(); i++) {
    out.push_back(x[i%x.size()]);
  }
  return out;
}

void TestGenericDerivative(const std::string& name, const std::vector<int>& xShape, const std::vector<int>& fShape,
    const MMFunction& f, const MMFunction& df, int nTrials, int nSubTrials) {

  DenseMMatrix x(xShape), outTmp(fShape), derivReal(Concat(fShape, xShape));

  bool had_error = false;
  for (int trials = 0; trials < nTrials; trials++) {
    RandMatrix(&x);

    df(&x, &derivReal); 

    for (int subtrials = 0; subtrials < nSubTrials; subtrials++ ) {
      std::vector<int>
        f_idx = RandIndex(fShape),
        x_idx = RandIndex(xShape);
      MMFloat approx = ApproxDerivative(f, &outTmp, &x, f_idx, x_idx);

      MMFloat real = derivReal.get(Concat(f_idx, x_idx));
    
      MMFloat approx_err = abs(real-approx);
      if (approx_err > 0.01) {
        std::cerr << name << " broken" << std::endl;
        std::cerr << "Trial: " << trials << "  Subtrial: " << subtrials << std::endl;
        std::cerr  << "f_idx: " << DebugString(f_idx) << std::endl;
        std::cerr << "x_idx: " << DebugString(x_idx) << std::endl;
        std::cerr << "x val: " << x.get(x_idx) << std::endl;
        std::cerr << "real:" << real << std::endl;
        std::cerr << "approx:" << approx << std::endl;
        had_error = true;
      }
    }
  }
  if (had_error) {
    std::exit(1);
  }
}


} // namespace test
} // namespace mmatrix
