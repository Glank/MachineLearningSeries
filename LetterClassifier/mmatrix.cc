#include "mmatrix.h"

#include <cmath>
#include <stdexcept>

namespace mmatrix {

using std::string;
using std::unique_ptr;
using std::vector;

bool Type::implies(const string& other_class_name) const {
  return other_class_name == class_name_;
}

float MMatrixInterface::get(int i) const {
  vector<int> indices(shape().size(), 0);
  FromValueIndex(shape(), i, &indices);
  return get(indices);
}
void MMatrixInterface::set(int i, float value) {
  vector<int> indices(shape().size(), 0);
  FromValueIndex(shape(), i, &indices);
  set(indices, value);
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
float DenseMMatrix::get(int i) const {
  return values_[i];
}
void DenseMMatrix::set(int i, float value) {
  values_[i] = value;
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
  if (a == nullptr || b == nullptr || out == nullptr) {
    throw std::invalid_argument("Multiply cannot take null arguments.");
  }
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
  for (int vl = 0; vl < lBound; vl++) {
    for (int vr = 0; vr < rBound; vr++) {
      float sum = 0;
      for (int vm = 0; vm < midBound; vm++) {
        int va = vl+vm*lBound;
        int vb = vm+vr*midBound;
        sum += a->get(va)*b->get(vb);
      }
      int vo = vl+vr*lBound;
      out->set(vo, sum);
    }
  }
}

bool AreEqual(const MMatrixInterface* a, const MMatrixInterface* b, float epsilon) {
  if (a == nullptr || b == nullptr) {
    throw std::invalid_argument("AreEqual cannot take null arguments.");
  }
  if (a->shape().size() != b->shape().size()) {
    throw std::out_of_range("Mismatched AreEqual shapes.");
  }
  int num_elements = 1;
  for (int i = 0; i < a->shape().size(); i++) {
    if (a->shape()[i] != b->shape()[i]) {
      throw std::out_of_range("Mismatched AreEqual shapes.");
    }
    num_elements*=a->shape()[i];
  }
  for (int i = 0; i < num_elements; i++) {
    if (std::abs(a->get(i)-b->get(i)) > epsilon) {
      return false;
    }
  }
  return true;
}

unique_ptr<MMatrixInterface> Ident(int n, const vector<int>& base_shape) {
  vector<int> shape(base_shape.size()*n, 0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < base_shape.size(); j++) {
      shape[i*base_shape.size()+j] = base_shape[j];
    }
  }
  int base_elements = 1;
  for (int i = 0; i < base_shape.size(); i++) {
    base_elements *= base_shape[i];
  }
  // TODO: use a sparse matrix instead
  unique_ptr<MMatrixInterface> ident(new DenseMMatrix(shape));
  for (int i = 0; i < base_elements; i++) {
    int index = i;
    int k = base_elements;
    for (int j = 1; j < n; j++) {
      index += k*i;
      k*=base_elements;
    }
    ident->set(index, 1);
  }
  return ident;
}

}  // namespace mmatrix
