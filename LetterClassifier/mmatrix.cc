#include "mmatrix.h"

#include <stdexcept>

namespace mmatrix {

using std::string;
using std::vector;

bool Type::implies(const string& other_class_name) const {
  return other_class_name == class_name_;
}

const Type MMatrixInterface::type_ = Type("::mmatrix::MMatrixInterface");
const Type* MMatrixInterface::type() const {
  return &type_;
}

DenseMMatrix::DenseMMatrix(const vector<int>& shape) : shape_(shape) {
  int total_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    total_size *= shape[i];
  }
  values_ = vector<float>(total_size, 0);
}
float DenseMMatrix::get(const vector<int>& indices) const {
  return values_[ToValueIndex(shape_, indices)];
}
void DenseMMatrix::set(const vector<int>& indices, float value) {
  values_[ToValueIndex(shape_, indices)] = value;
}
const vector<int>& DenseMMatrix::shape() const {
  return shape_;
}

int ToValueIndex(const vector<int>& shape, const vector<int>& indices) {
  int vindex = 0;
  int multiplier = 1;
  for (int i = 0; i < shape.size(); i++) {
    if (indices[i] < 0 || indices[i] >= shape[i]) {
      throw std::out_of_range("ToValueIndex out of range.");
    }
    vindex += indices[i] * multiplier;
    multiplier *= shape[i];
  }
  return vindex;
}
void FromValueIndex(const vector<int>& shape, int vindex, vector<int>* indices) {
  for (int i = 0; i < shape.size(); i++) {
    (*indices)[i] = vindex % shape[i];
    vindex = vindex/shape[i];
  }
}

void Multiply(int n, const MMatrixInterface* a, const MMatrixInterface* b,
    MMatrixInterface* out) {
  // ||a|| = l + n
  // ||b|| = n + r
  int l = a->shape().size()-n;
  int r = b->shape().size()-n;
  if (l < 0 || r < 0) {
    throw std::out_of_range("Invalid multiplication size");
  }
  int midBound = 1;
  for (int i = 0; i < n; i++) {
    if (a->shape()[l+i] != b->shape()[i]) {
      throw std::out_of_range("Invalid multiplication shapes.");
    }
    midBound *= b->shape()[i];
  }
  int lBound  = 1;
  for (int i = 0; i < l; i++) {
    if (a->shape()[i] != out->shape()[i]) {
      throw std::out_of_range("Invalid left output multiplication shape.");
    }
    lBound *= a->shape()[i];
  }
  int rBound = 1;
  for (int i = 0; i < r; i++) {
    if (b->shape()[n+i] != out->shape()[l+i]) {
      throw std::out_of_range("Invalid right output multiplication shape.");
    }
    rBound *= b->shape()[n+i];
  }
  vector<int> aIndices(a->shape().size(), 0);
  vector<int> bIndices(b->shape().size(), 0);
  vector<int> outIndices(out->shape().size(), 0);
  for (int vl = 0; vl < lBound; vl++) {
    for (int vr = 0; vr < rBound; vr++) {
      float sum = 0;
      for (int vm = 0; vm < midBound; vm++) {
        int va = vl+vm*lBound;
        int vb = vm+vr*midBound;
        FromValueIndex(a->shape(), va, &aIndices);
        FromValueIndex(b->shape(), vb, &bIndices);
        sum += a->get(aIndices)*b->get(bIndices);
      }
      int vo = vl+vr*lBound;
      FromValueIndex(out->shape(), vo, &outIndices);
      out->set(outIndices, sum);
    }
  }
}

}  // namespace mmatrix
