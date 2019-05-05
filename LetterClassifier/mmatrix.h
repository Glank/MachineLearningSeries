#ifndef MMATRIX_H
#define MMATRIX_H

#include <string>
#include <vector>

namespace mmatrix {

class Type {
 public:
  // Class names must be unique to the class of matrix.
  Type(const std::string& class_name) : class_name_(class_name) {}
  virtual ~Type() = default;
  const std::string& className() const { return class_name_; }
  // Returns true if this matrix can be cast to an instance of the given other type.
  virtual bool implies(const std::string& other_class_name) const;
 private:
  std::string class_name_;
};

class MMatrixInterface {
 public:
  virtual ~MMatrixInterface() = default;
  // Returns the value stored at the given indices.
  virtual float get(const std::vector<int>& indices) const = 0;
  // Stores a value at the given indicies. Each index must be within it's
  // dimension defined by shape().
  virtual void set(const std::vector<int>& indices, float value) = 0;
  // Returns the shape (size of dimensions) of this matrix.
  virtual const std::vector<int>& shape() const = 0;
  // Returns the most generic type "::mmatrix::MMatrixInterface" by default.
  virtual const Type* type() const;
 private:
  const static Type type_;
};

class DenseMMatrix : public MMatrixInterface {
 public:
  DenseMMatrix(const std::vector<int>& shape);
  ~DenseMMatrix() = default;

  float get(const std::vector<int>& indices) const override;
  void set(const std::vector<int>& indices, float value) override;
  const std::vector<int>& shape() const override;

 private:
  std::vector<float> values_;
  std::vector<int> shape_;
};

int ToValueIndex(const std::vector<int>& shape, const std::vector<int>& indices);
void FromValueIndex(const std::vector<int>& shape, int vindex,
    std::vector<int>* indices);

void Multiply(int n, const MMatrixInterface* a,
  const MMatrixInterface* b, MMatrixInterface* out);

}  // namespace mmatrix

#endif
