#ifndef MMATRIX_TEST_UTILS_H
#define MMATRIX_TEST_UTILS_H

#include "mmatrix.h"

#include <vector>

namespace mmatrix {
namespace test {

MMFloat randf();

std::vector<int> RandShape(int length = -1 /* rand 0 to 2 */, int max = 5);

std::vector<int> RandIndex(const std::vector<int>& shape);

void RandMatrix(MMatrixInterface* out, MMFloat zerochance = 0, int min = -5, int max=5);

// Approximate df/dx(x)[f_index + x_index]
MMFloat ApproxDerivative(MMFunction f, MMatrixInterface* out_tmp,
    MMatrixInterface* x, std::vector<int> f_index, std::vector<int> x_index, MMFloat epsilon = 1e-6);

std::vector<int> ConcatNTimes(int n, const std::vector<int>& x);

void TestGenericDerivative(const std::string& name, const std::vector<int>& xShape, const std::vector<int>& fShape,
    const MMFunction& f, const MMFunction& df, int nTrials, int nSubTrials);

} // namespace test
} // namespace mmatrix

#endif // MMATRIX_TEST_UTILS_H
