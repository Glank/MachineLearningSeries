#ifndef MMATRIX_H
#define MMATRIX_H

#include <unordered_map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace mmatrix {

namespace internal {

typedef uint64_t MMatrixType;
constexpr MMatrixType kMMatrixType_Dense = 0;
constexpr MMatrixType kMMatrixType_Sparse = 1;

}  // namespace internal

class MMatrixInterface {
 public:
  virtual ~MMatrixInterface() = default;
  // Returns the value stored at the given indices.
  virtual float get(const std::vector<int>& indices) const;
  // Returns the value stored at the given indices represented by i.
  // See ToValueIndex.
  virtual float get(int i) const;
  // Stores a value at the given indicies represented by i.
  // See ToValueIndex.
  virtual void set(int i, float value);
  // Stores a value at the given indicies. Each index must be within it's
  // dimension defined by shape().
  virtual void set(const std::vector<int>& indices, float value);
  // Returns the shape (size of dimensions) of this matrix.
  virtual const std::vector<int>& shape() const = 0;
  // Returns the most specific class type of the MMatrix.
  virtual internal::MMatrixType type() const = 0;
  // Sets all values of the matrix to zero.
  virtual void zero() = 0;
};

class DenseMMatrix : public MMatrixInterface {
 public:
  DenseMMatrix(const std::vector<int>& shape);
  ~DenseMMatrix() = default;

  using MMatrixInterface::get;
  using MMatrixInterface::set;
  float get(int i) const override;
  void set(int i, float value) override;
  const std::vector<int>& shape() const override;
  void zero() override;
  internal::MMatrixType type() const override;

 private:
  std::vector<float> values_;
  std::vector<int> shape_;
};

class SparseMMatrix : public MMatrixInterface {
 public:
  SparseMMatrix(const std::vector<int>& shape);
  ~SparseMMatrix() = default;

  using MMatrixInterface::get;
  using MMatrixInterface::set;
  float get(int i) const override;
  void set(int i, float value) override;
  const std::vector<int>& shape() const override;
  void zero() override;
  internal::MMatrixType type() const override;

  // Returns the number of non-zero values in the matrix.
  int size() const;

  std::unordered_map<int, float>::iterator begin() { return values_.begin(); }
  std::unordered_map<int, float>::iterator end() { return values_.end(); }
  std::unordered_map<int, float>::const_iterator cbegin() { return values_.cbegin(); }
  std::unordered_map<int, float>::const_iterator cend() { return values_.cend(); }
  std::unordered_map<int, float>::const_iterator begin() const { return values_.cbegin(); }
  std::unordered_map<int, float>::const_iterator end() const { return values_.cend(); }
 private:
  // Maps the value index to non-zero values.
  std::unordered_map<int, float> values_;

  std::vector<int> shape_;
};

int ToValueIndex(const std::vector<int>& shape, const std::vector<int>& indices);
void FromValueIndex(const std::vector<int>& shape, int vindex,
    std::vector<int>* indices);

void Multiply(int n, const MMatrixInterface* a,
  const MMatrixInterface* b, MMatrixInterface* out);

void AddTo(const MMatrixInterface* m, MMatrixInterface* out);

// Throws an error when matricies aren't even the same shape.
// Epsilon is an optional small positive value under which differences are
// still considered "equal". So, if epsilon=0.0001, then technically different
// matricies a and b will still be considered equal if none of their elements
// differ by more than 0.0001.
bool AreEqual(const MMatrixInterface* a, const MMatrixInterface* b, float epsilon = 0);

// Returns an n'th order identity with the given base_shape.
std::unique_ptr<MMatrixInterface> Ident(int n, const std::vector<int>& base_shape);

}  // namespace mmatrix

#endif
