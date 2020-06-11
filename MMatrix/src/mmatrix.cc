#include "mmatrix.h"

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <sstream>

namespace mmatrix {

using std::string;
using std::unique_ptr;
using std::vector;

MMFloat MMatrixInterface::get(int i) const {
  vector<int> indices(shape().size(), 0);
  FromValueIndex(shape(), i, &indices);
  return get(indices);
}
void MMatrixInterface::set(int i, MMFloat value) {
  vector<int> indices(shape().size(), 0);
  FromValueIndex(shape(), i, &indices);
  set(indices, value);
}
MMFloat MMatrixInterface::get(const vector<int>& indices) const {
  return get(ToValueIndex(shape(), indices));
}
void MMatrixInterface::set(const vector<int>& indices, MMFloat value) {
  set(ToValueIndex(shape(), indices), value);
}

DenseMMatrix::DenseMMatrix(const vector<int>& shape) : shape_(shape) {
  int total_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    total_size *= shape[i];
  }
  values_ = vector<MMFloat>(total_size, 0);
}
MMFloat DenseMMatrix::get(int i) const {
  return values_[i];
}
void DenseMMatrix::set(int i, MMFloat value) {
  values_[i] = value;
}
const vector<int>& DenseMMatrix::shape() const {
  return shape_;
}
void DenseMMatrix::zero() {
  for (int i = 0; i < values_.size(); i++) {
    values_[i] = 0;
  }
}
internal::MMatrixType DenseMMatrix::type() const {
  return internal::kMMatrixType_Dense;
}

SparseMMatrix::SparseMMatrix(const std::vector<int>& shape) : shape_(shape) {}
MMFloat SparseMMatrix::get(int i) const {
  const auto v = values_.find(i);
  if (v != values_.end()) {
    return v->second;
  }
  return 0;
}
void SparseMMatrix::set(int i, MMFloat value) {
  if (value == 0) {
    const auto it = values_.find(i);
    if (it != values_.end()) {
      values_.erase(it);
    }
  } else {
    values_[i] = value;
  }
}
const std::vector<int>& SparseMMatrix::shape() const {
  return shape_;
}
internal::MMatrixType SparseMMatrix::type() const {
  return internal::kMMatrixType_Sparse;
}
void SparseMMatrix::zero() {
  values_.clear();
}
int SparseMMatrix::size() const {
  return values_.size();
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

namespace {
void MultiplySparseAB(int n, const SparseMMatrix* a, const SparseMMatrix* b,
    MMatrixInterface* out, int lBound, int midBound, int rBound) {
  //std::cout << "AB" << std::endl;

  int a_index_ops = a->size()*rBound;
  int b_index_ops = b->size()*lBound;

  out->zero();
  if (a_index_ops <= b_index_ops) {
    for (const auto& a_pair : *a) {
      int vl = a_pair.first % lBound;
      int vm = a_pair.first / lBound;
      for (int vr = 0; vr < rBound; vr++) {
        int vo = vl+vr*lBound;
        int vb = vm+vr*midBound;
        out->set(vo, out->get(vo) + a_pair.second*b->get(vb));
      }
    }
  } else {
    for (const auto& b_pair : *b) {
      int vm = b_pair.first % midBound;
      int vr = b_pair.first / midBound;
      for (int vl = 0; vl < lBound; vl++) {
        int va = vl+vm*lBound;
        int vo = vl+vr*lBound;
        out->set(vo, out->get(vo) + a->get(va)*b_pair.second);
      }
    }
  }
}
void MultiplySparseA(int n, const SparseMMatrix* a, const MMatrixInterface* b,
  MMatrixInterface* out, int lBound, int midBound, int rBound) {
  //std::cout << "A" << std::endl;
  out->zero();
  for (const auto& a_pair : *a) {
    int vl = a_pair.first % lBound;
    int vm = a_pair.first / lBound;
    for (int vr = 0; vr < rBound; vr++) {
      int vo = vl+vr*lBound;
      int vb = vm+vr*midBound;
      out->set(vo, out->get(vo) + a_pair.second*b->get(vb));
    }
  }
}
void MultiplySparseB(int n, const MMatrixInterface* a, const SparseMMatrix* b,
  MMatrixInterface* out, int lBound, int midBound, int rBound) {
  //std::cout << "B" << std::endl;
  out->zero();
  for (const auto& b_pair : *b) {
    int vm = b_pair.first % midBound;
    int vr = b_pair.first / midBound;
    for (int vl = 0; vl < lBound; vl++) {
      int va = vl+vm*lBound;
      int vo = vl+vr*lBound;
      out->set(vo, out->get(vo) + a->get(va)*b_pair.second);
    }
  }
}
}  // namespace

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

  // Cast to more specific types if applicable to speed up execution.
  if (a->type() == internal::kMMatrixType_Sparse
      && b->type() == internal::kMMatrixType_Sparse) {
    MultiplySparseAB(n, static_cast<const SparseMMatrix*>(a),
      static_cast<const SparseMMatrix*>(b), out,
      lBound, midBound, rBound);
    return;
  } else if (a->type() == internal::kMMatrixType_Sparse) {
    MultiplySparseA(n, static_cast<const SparseMMatrix*>(a), b, out,
      lBound, midBound, rBound);
    return;
  } else if (b->type() == internal::kMMatrixType_Sparse) {
    MultiplySparseB(n, a, static_cast<const SparseMMatrix*>(b), out,
      lBound, midBound, rBound);
    return;
  }

  for (int vl = 0; vl < lBound; vl++) {
    for (int vr = 0; vr < rBound; vr++) {
      MMFloat sum = 0;
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

void AddTo(const MMatrixInterface* m, MMatrixInterface* out) {
  if (m == nullptr || out == nullptr) {
    throw std::invalid_argument("AddTo cannot take null arguments.");
  }

  // bound = product of m's shape
  int bound = 1;
  for (int s : m->shape()) {
    bound *= s;
  }
  
  for (int i = 0; i < bound; i++) {
    out->set(i, out->get(i)+m->get(i));
  }
}

void Elementwise(std::function<MMFloat(MMFloat)> f, const MMatrixInterface* m,
    MMatrixInterface* out) {
  if (m == nullptr || out == nullptr) {
    throw std::invalid_argument("Elementwise cannot take null arguments.");
  }

  int bound = 1;
  for (int s : m->shape()) {
    bound *= s;
  }

  // TODO: handle sparse matrixies better
  for (int i = 0; i < bound; i++) {
    out->set(i, f(m->get(i)));
  }
}

void Transpose(int n, const MMatrixInterface* m, MMatrixInterface* out) {
  if (m == nullptr || out == nullptr) {
    throw std::invalid_argument("Transpose cannot take null arguments.");
  }

  int lBound = 1;
  int rBound = 1;
  for (int i = 0; i < n; i++) {
    lBound *= m->shape().at(i);
  }
  for (int i = n; i < m->shape().size(); i++) {
    rBound *= m->shape().at(i);
  }

  // TODO: handle sparse matrixies better
  for(int l = 0; l < lBound; l++) {
    for (int r = 0; r < rBound; r++) {
      int im = l+r*lBound;
      int io = r+l*rBound;
      out->set(io, m->get(im));
    }
  }
}

bool AreEqual(const MMatrixInterface* a, const MMatrixInterface* b, MMFloat epsilon) {
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
  unique_ptr<MMatrixInterface> ident(new SparseMMatrix(shape));
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

std::vector<int> Concat(const std::vector<int>& a, const std::vector<int>& b) {
  std::vector<int> out;
  out.reserve(a.size()+b.size());
  out.insert(out.end(), a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());
  return out;
}
std::vector<int> Concat(const std::vector<int>& a, const std::vector<int>& b, const std::vector<int>& c) {
  std::vector<int> out;
  out.reserve(a.size()+b.size());
  out.insert(out.end(), a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());
  out.insert(out.end(), c.begin(), c.end());
  return out;
}

std::string DebugString(const MMatrixInterface* m) {
  std::stringstream tmp;

  tmp << '<';
  int ncells = 1;
  for (int i = 0; i < m->shape().size(); i++) {
    if (i != 0) {
      tmp << ", ";
    }
    tmp << m->shape()[i];
    ncells *= m->shape()[i];
  }
  tmp << ">" << std::endl;
  
  for (int i = 0; i < ncells; i++) {
    if (i != 0) {
      if (i%20 == 0 && i != ncells-1)
        tmp << "," << std::endl; 
      else
        tmp << ", ";
    }
    tmp << m->get(i);
  }

  return tmp.str();
}

std::string DebugString(const std::vector<int>& v) {
  std::stringstream tmp;
  tmp << '<';
  for (int i = 0; i < v.size(); i++) {
    if (i != 0) {
      tmp << ", ";
    }
    tmp << v[i];
  }
  tmp << ">";
  return tmp.str();
}

void Copy(const MMatrixInterface* m, MMatrixInterface* out) {
  // TODO: improve for sparse matricies
  if (m->shape().size() != out->shape().size()) {
    throw std::invalid_argument("Wrong shape for copy.");
  }
  int ncells = 1;
  for (int i = 0; i < m->shape().size(); i++) {
    if (m->shape()[i] != out->shape()[i]) {
      throw std::invalid_argument("Wrong shape for copy.");
    }
    ncells *= m->shape()[i];
  }
  for (int i = 0; i < ncells; i++) {
    out->set(i, m->get(i));
  }
}

}  // namespace mmatrix
