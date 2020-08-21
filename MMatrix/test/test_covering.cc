#include "mmatrix.h"
#include "covering.h"
#include "mmatrix_test_utils.h"

#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>
#include <functional>

using namespace mmatrix;
using namespace mmatrix::test;

void TestCoveringErrorDerivative() {
  std::cout << "Testing Covering Error Derivative..." << std::endl;
  std::srand(31415);
  std::vector<int> xShape = {3, 10};
  std::vector<int> fShape = {};

  std::unique_ptr<DenseMMatrix> T(new DenseMMatrix(xShape));
  RandMatrix(T.get(), 0, 0, 1000);
  covering::CoveringMatrices covering(std::move(T));
  covering.set_primary_params(6);

  MMFunction covErr = [&](const MMatrixInterface* x, MMatrixInterface* out) {
    covering.update_weights([&](MMatrixInterface* w) { Copy(x, w);});
    out->set(0, covering.primary());
  };

  MMFunction covErrDeriv = [&](const MMatrixInterface* x, MMatrixInterface* out) {
    covering.update_weights([&](MMatrixInterface* w) { Copy(x, w);});
    Copy(covering.primary_deriv(), out);
  };

  TestGenericDerivative("Covering Error", xShape, fShape, covErr, covErrDeriv, 10, 10);
}

void TestCoveringSigmoidDerivative() {
  std::cout << "Testing Covering Sigmoid Error Derivative..." << std::endl;
  std::srand(31416);
  std::vector<int> xShape = {3, 10};
  std::vector<int> fShape = {3, 10};

  covering::CoveringMatrices covering(std::unique_ptr<SparseMMatrix>(new SparseMMatrix(xShape)));
  MMFunction sig = [&](const MMatrixInterface* x, MMatrixInterface* out) {
    covering.update_weights([&](MMatrixInterface* w) { Copy(x, w);});
    Copy(covering.sigmoid(), out);
  };

  MMFunction sigDeriv = [&](const MMatrixInterface* x, MMatrixInterface* out) {
    covering.update_weights([&](MMatrixInterface* w) { Copy(x, w);});
    Copy(covering.sigmoid_deriv(), out);
  };

  TestGenericDerivative("Covering Sigmoid Error", xShape, fShape, sig, sigDeriv, 10, 5);
}

void TestCoveringDensityDerivative() {
  std::cout << "Testing Covering Density Derivative..." << std::endl;
  std::srand(31415);
  std::vector<int> xShape = {3, 10};
  std::vector<int> fShape = {};

  covering::CoveringMatrices covering(std::unique_ptr<SparseMMatrix>(new SparseMMatrix(xShape)));

  MMFunction densityErr = [&](const MMatrixInterface* x, MMatrixInterface* out) {
    covering.update_weights([&](MMatrixInterface* w) { Copy(x, w);});
    out->set(0, covering.density());
  };

  MMFunction densityErrDeriv = [&](const MMatrixInterface* x, MMatrixInterface* out) {
    covering.update_weights([&](MMatrixInterface* w) { Copy(x, w);});
    Copy(covering.density_deriv(), out);
  };

  TestGenericDerivative("Covering Density", xShape, fShape, densityErr, densityErrDeriv, 10, 10);
}

void TestCoveringContinuityDerivative() {
  std::cout << "Testing Covering Continuity Derivative..." << std::endl;
  std::srand(31415);
  std::vector<int> xShape = {3, 10};
  std::vector<int> fShape = {};

  covering::CoveringMatrices covering(std::unique_ptr<SparseMMatrix>(new SparseMMatrix(xShape)));

  MMFunction continuityErr = [&](const MMatrixInterface* w, MMatrixInterface* out) {
    covering.update_weights([&](MMatrixInterface* weights) { Copy(w, weights); });
    out->set(0, covering.continuity());
  };

  MMFunction continuityErrDeriv = [&](const MMatrixInterface* w, MMatrixInterface* out) {
    covering.update_weights([&](MMatrixInterface* weights) { Copy(w, weights); });
    Copy(covering.continuity_deriv(), out);
  };

  TestGenericDerivative("Covering Continuity", xShape, fShape, continuityErr, continuityErrDeriv, 10, 10);
}

void TestCoveringRegulationDerivative() {
  std::cout << "Testing Covering Regulation Derivative..." << std::endl;
  std::srand(31415);
  std::vector<int> xShape = {3, 10};
  std::vector<int> fShape = {};

  covering::CoveringMatrices covering(std::unique_ptr<SparseMMatrix>(new SparseMMatrix(xShape)));

  MMFunction regErr = [&](const MMatrixInterface* w, MMatrixInterface* out) {
    covering.update_weights([&](MMatrixInterface* weights) { Copy(w, weights); });
    out->set(0, covering.regulation());
  };

  MMFunction regErrDeriv = [&](const MMatrixInterface* w, MMatrixInterface* out) {
    covering.update_weights([&](MMatrixInterface* weights) { Copy(w, weights); });
    Copy(covering.regulation_deriv(), out);
  };

  TestGenericDerivative("Covering Regulation", xShape, fShape, regErr, regErrDeriv, 10, 10);
}

int main() {
  TestCoveringErrorDerivative();
  TestCoveringSigmoidDerivative();
  TestCoveringDensityDerivative();
  TestCoveringContinuityDerivative();
  TestCoveringRegulationDerivative();
  std::cout << "All tests pass." << std::endl;
  return 0;
}
