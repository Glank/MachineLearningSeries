#include "mmatrix.h"

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <sstream>

namespace mmatrix {

using std::string;
using std::unique_ptr;
using std::vector;

namespace {

inline double product(vector<int> shape) {
  long prod = 1;
  for (long s : shape) {
    prod *= s;
  }
  return prod;
}

}  // namespace

MMFloat MMatrixInterface::get(long i) const {
  vector<int> indices(shape().size(), 0);
  FromValueIndex(shape(), i, &indices);
  return get(indices);
}
void MMatrixInterface::set(long i, MMFloat value) {
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
MMFloat DenseMMatrix::get(long i) const {
  return values_[static_cast<int>(i)];
}
void DenseMMatrix::set(long i, MMFloat value) {
  values_[static_cast<int>(i)] = value;
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

SparseMMatrix::SparseMMatrix(const std::vector<int>& shape) : shape_(shape), size_(product(shape)) {}
MMFloat SparseMMatrix::get(long i) const {
  const auto v = values_.find(i);
  if (v != values_.end()) {
    return v->second;
  }
  return 0;
}
void SparseMMatrix::set(long i, MMFloat value) {
  if (i < 0 || i >= size_) {
    std::cerr << "0 <= " << i << " < " << size_ << std::endl;
    throw std::out_of_range("Invalid index for sparse mmatrix.");
  }
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

long ToValueIndex(const vector<int>& shape, const vector<int>& indices) {
  long vindex = 0;
  long multiplier = 1;
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

void HMultiply(int n, const MMatrixInterface* a, const MMatrixInterface* b, MMatrixInterface* out) {
  if (a == nullptr || b == nullptr || out == nullptr) {
    throw std::invalid_argument("HMultiply cannot take null arguments.");
  }

  // ||a|| = l + n
  // ||b|| = n + r
  // ||a * b|| = l + n + r
  int l = a->shape().size()-n;
  int r = b->shape().size()-n;
  if (l < 0 || r < 0) {
    throw std::out_of_range("Invalid Hadamard multiplication size");
  }
  int midBound = 1;
  for (int i = 0; i < n; i++) {
    if (a->shape()[l+i] != b->shape()[i]) {
      throw std::out_of_range("Invalid Hadamard multiplication shapes.");
    }
    if (a->shape()[l+i] != out->shape()[l+i]) {
      throw std::out_of_range("Invalid mid output Hadamard multiplication shape.");
    }
    midBound *= b->shape()[i];
  }
  int lBound  = 1;
  for (int i = 0; i < l; i++) {
    if (a->shape()[i] != out->shape()[i]) {
      throw std::out_of_range("Invalid left output Hadamard multiplication shape.");
    }
    lBound *= a->shape()[i];
  }
  int rBound = 1;
  for (int i = 0; i < r; i++) {
    if (b->shape()[n+i] != out->shape()[l+n+i]) {
      throw std::out_of_range("Invalid right output Hadamard multiplication shape.");
    }
    rBound *= b->shape()[n+i];
  }

  // TODO: accelerate for sparse matricies
  for (int i = 0; i < lBound; i++) {
    for (int j = 0; j < midBound; j++) {
      int va = i+lBound*j;
      for (int k = 0; k < rBound; k++) {
        int vo = i+lBound*j+lBound*midBound*k;
        int vb = j+midBound*k;
        out->set(vo, a->get(va)*b->get(vb)); 
      }
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

void SubFrom(const MMatrixInterface* m, MMatrixInterface* out) {
  if (m == nullptr || out == nullptr) {
    throw std::invalid_argument("SubFrom cannot take null arguments.");
  }

  // bound = product of m's shape
  int bound = 1;
  for (int s : m->shape()) {
    bound *= s;
  }
  
  for (int i = 0; i < bound; i++) {
    out->set(i, out->get(i)-m->get(i));
  }
}

void Combine(const std::vector<const MMatrixInterface*>& matrices, const std::vector<MMFloat>& consts, MMatrixInterface* out) {
  bool any_null = false;
  for (const MMatrixInterface* m : matrices) {
    if(m == nullptr) {
      any_null = true;
      break;
    }
  }
  if (out == nullptr || any_null) {
    throw std::invalid_argument("Combine cannot take null arguments.");
  }
  if (matrices.size() != consts.size()) {
    throw std::invalid_argument("Combine requires same number of consts as matrices.");
  }

  if (matrices.empty()) {
    out->zero();
    return;
  }
  
  int bound = 1;
  for (int s : matrices[0]->shape()) {
    bound *= s;
  }

  for (int i = 0; i < bound; i++) {
    double term = 0;
    for (int m = 0; m < matrices.size(); m++) {
      term += consts[m]*matrices[m]->get(i);
    }
    out->set(i, term);
  }
}

double Sum(const MMatrixInterface* m) {
  if (m == nullptr) {
    throw std::invalid_argument("Sum cannot take null arguments.");
  }

  // bound = product of m's shape
  int bound = 1;
  for (int s : m->shape()) {
    bound *= s;
  }
  
  double result = 0;
  for (int i = 0; i < bound; i++) {
    result += m->get(i);
  }
  return result;
}

double SquaredSum(const MMatrixInterface* m) {
  if (m == nullptr) {
    throw std::invalid_argument("SquaredSum cannot take null arguments.");
  }

  // bound = product of m's shape
  int bound = 1;
  for (int s : m->shape()) {
    bound *= s;
  }
  
  double result = 0;
  for (int i = 0; i < bound; i++) {
    double tmp = m->get(i);
    result += tmp*tmp;
  }
  return result;
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

void Fill(MMFloat f, MMatrixInterface* m) {
  if (m == nullptr) {
    throw std::invalid_argument("Fill cannot take null argument.");
  }

  int bound = 1;
  for (int s : m->shape()) {
    bound *= s;
  }

  for (int i = 0; i < bound; i++) {
    m->set(i, f);
  }
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

void Reshape(const MMatrixInterface* m, MMatrixInterface* out) {
  // TODO: improve for sparse matrices
  // TODO: shape checks
  int ncells = 1;
  for (int i = 0; i < m->shape().size(); i++) {
    ncells *= m->shape()[i];
  }
  for (int i = 0; i < ncells; i++) {
    out->set(i, m->get(i));
  }
}

bool Invert(int n, MMatrixInterface* m, MMatrixInterface* out) {
  // The size of m is N*N
  int N;
  {
    if (n > m->shape().size()) {
      throw std::runtime_error("Invalid n in Invert");
    }
    long N1 = 1;
    long N2 = 1;
    int i = 0;
    for(i = 0; i < n; i++) {
      N1 *= m->shape()[i];
    }
    for(; i < m->shape().size(); i++){
      N2 *= m->shape()[i];
    }
    if (N1 != N2) {
      throw std::runtime_error("Invalid n in Invert");
    }
    if (N1 > std::numeric_limits<int>::max()) {
      throw std::runtime_error("MMatrix too large to invert.");
    }
    N = static_cast<int>(N1);
  }


  // Set up NxN inverse
  out->zero();
  for (int i = 0; i < N; i++) {
    out->set(i+N*i, 1);
  }

  // Put m in reduced row-echelon form.
  int pr = 0;
  for (int pc = 0; pc < N; pc++) {

    /* Debug print
    for (int r = 0; r < N; r++) {
      for (int c = 0; c < N; c++) {
        std::cout << m->get(r+N*c) << "  ";
      }
      std::cout << "|  ";
      for (int c = 0; c < N; c++) {
        std::cout << out->get(r+N*c) << "  ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
    // */

    // If the current head is zero, find the next row at pr or below
    // that is non-zero and swap it.
    if (m->get(pr+N*pc) == 0) {
      int r = pr+1;
      while(m->get(r+N*pc) == 0 && r < N) r++;
      if (r==N) continue;
      // swap
      double tmp;
      for (int c = pc; c < N; c++) {
        tmp = m->get(r+N*c);
        m->set(r+N*c, m->get(pr+N*c));
        m->set(pr+N*c, tmp);
      }
      for (int c = 0; c < N; c++) {
        tmp = out->get(r+N*c);
        out->set(r+N*c, out->get(pr+N*c));
        out->set(pr+N*c, tmp);
      }
    }

    // Divide that row by it's pivot
    double pivot = m->get(pr+N*pc);
    m->set(pr+N*pc, 1);
    for (int c = pc+1; c < N; c++)
      m->set(pr+N*c, m->get(pr+N*c)/pivot);
    for (int c = 0; c < N; c++)
      out->set(pr+N*c, out->get(pr+N*c)/pivot);

    // Clear the pivot column below the pivot by subracting
    // multiples of the pivot row from each row below.
    for (int r = pr+1; r<N; r++) {
      double head = m->get(r+N*pc);
      if (head == 0) continue;
      for (int c = pc; c < N; c++)
        m->set(r+N*c, m->get(r+N*c) - m->get(pr+N*c)*head);
      for (int c = 0; c < N; c++)
        out->set(r+N*c, out->get(r+N*c) - out->get(pr+N*c)*head);
    }

    // Increment the pivot row
    pr++;
  }

  // Verify that the RRE form is invertable
  for (int i = 0; i < N; i++) {
    if (m->get(i+N*i) != 1) {
      return false;
    }
  }

  /* Debug print
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < N; c++) {
      std::cout << m->get(r+N*c) << "  ";
    }
    std::cout << "|  ";
    for (int c = 0; c < N; c++) {
      std::cout << out->get(r+N*c) << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl << std::endl;
  // */

  // Clear the columns above the pivots
  pr = N-1;
  for (int pc = N-1; pc >= 0; pc--) {
    // Go up until we find a non-zero element in the pivot column
    while(m->get(pr+N*pc) == 0 && pr >= 0) pr--;
    if (pr <= 0) break;

    // Subtract this row every row above it.
    for (int r = 0; r < pr; r++) {
      double tail = m->get(r+N*pc);
      m->set(r+N*pc, 0);
      for (int c = 0; c < N; c++)
        out->set(r+N*c, out->get(r+N*c) - out->get(pr+N*c)*tail);
    }

    /* Debug print
    for (int r = 0; r < N; r++) {
      for (int c = 0; c < N; c++) {
        std::cout << m->get(r+N*c) << "  ";
      }
      std::cout << "|  ";
      for (int c = 0; c < N; c++) {
        std::cout << out->get(r+N*c) << "  ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
    // */
    
    // Go up to the next pivot row
    pr--;
  }
  return true;
}

}  // namespace mmatrix
