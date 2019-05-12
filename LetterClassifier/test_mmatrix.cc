#include "mmatrix.h"

#include <iostream>
#include <cstdlib>
#include <vector>

using namespace mmatrix;
using std::vector;
using std::unique_ptr;

void TestToFromVIndex() {
  std::cout << "TestToFromVIndex..." << std::endl;
  vector<int> shape = {7,5,3};
  int vi = ToValueIndex(shape, {5,3,1});
  vector<int> indices(3, 0);
  FromValueIndex(shape, vi, &indices);
  if (indices[0] != 5) {
    std::cerr << "TestToFromVIndex Err at 0 index." << std::endl;
    std::exit(1);
  }
  if (indices[1] != 3) {
    std::cerr << "TestToFromVIndex Err at 1 index." << std::endl;
    std::exit(1);
  }
  if (indices[2] != 1) {
    std::cerr << "TestToFromVIndex Err at 2 index." << std::endl;
    std::exit(1);
  }
}

void TestDenseMMatrixGetSet() {
  std::cout << "TestDenseMMatrixGetSet..." << std::endl;
  DenseMMatrix m({3,3,2});
  m.set({0,0,0}, 1);
  m.set({0,0,1}, 2);
  m.set({0,1,0}, 3);
  m.set({0,1,1}, 4);
  m.set({0,2,0}, 5);
  m.set({0,2,1}, 6);

  m.set({1,0,0}, 7);
  m.set({1,0,1}, 8);
  m.set({1,1,0}, 9);
  m.set({1,1,1}, 10);
  m.set({1,2,0}, 11);
  m.set({1,2,1}, 12);

  m.set({2,0,0}, 13);
  m.set({2,0,1}, 14);
  m.set({2,1,0}, 15);
  m.set({2,1,1}, 16);
  m.set({2,2,0}, 17);
  m.set({2,2,1}, 18);

  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      for (int c = 0; c < 2; c++) {
        float expected = a*6+b*2+c+1;
        if (m.get({a,b,c}) != expected) {
          std::cerr << "Unexpected result in TestDenseMMatrixGetSet" << std::endl;
          std::exit(0);
        }
      }
    }
  }
}

void TestMMatrixMultiplication() {
  std::cout << "TestMMatrixMultiplication..." << std::endl;

  DenseMMatrix a({2,2});
  a.set({0,0}, 2);
  a.set({0,1}, 0);
  a.set({1,0}, 0);
  a.set({1,1}, 2);
  
  DenseMMatrix b({2,3});
  b.set({0,0}, 1);
  b.set({0,1}, 2);
  b.set({0,2}, 3);
  b.set({1,0}, 4);
  b.set({1,1}, 5);
  b.set({1,2}, 6);

  DenseMMatrix out({2,3});
  Multiply(1, &a, &b, &out);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      if (out.get({i,j}) != b.get({i,j})*2) {
        std::cerr << "Error with multiplication." << std::endl;
        std::exit(1);
      }
    }
  }
}

void TestMMatrixMultiplicationAssociation() {
  std::cout << "TestMMatrixMultiplicationAssociation..." << std::endl;

  DenseMMatrix a({2,2,2,2});
  a.set({0,0,0,0}, 0);
  a.set({0,0,0,1}, 1);
  a.set({0,0,1,0}, 0);
  a.set({0,0,1,1}, 0);
  a.set({0,1,0,0}, 0);
  a.set({0,1,0,1}, 0);
  a.set({0,1,1,0}, 0);
  a.set({0,1,1,1}, 0);
  a.set({1,0,0,0}, 0);
  a.set({1,0,0,1}, 0);
  a.set({1,0,1,0}, 0);
  a.set({1,0,1,1}, 0);
  a.set({1,1,0,0}, 0);
  a.set({1,1,0,1}, 0);
  a.set({1,1,1,0}, 0);
  a.set({1,1,1,1}, 0);

  DenseMMatrix b({2,2,2});
  b.set({0,0,0}, 0);
  b.set({0,0,1}, 0);
  b.set({0,1,0}, 1);
  b.set({0,1,1}, 0);
  b.set({1,0,0}, 0);
  b.set({1,0,1}, 0);
  b.set({1,1,0}, 0);
  b.set({1,1,1}, 0);
  
  DenseMMatrix c({2,2,2,2});
  c.set({0,0,0,0}, 1);
  c.set({0,0,0,1}, 0);
  c.set({0,0,1,0}, 0);
  c.set({0,0,1,1}, 0);
  c.set({0,1,0,0}, 0);
  c.set({0,1,0,1}, 0);
  c.set({0,1,1,0}, 0);
  c.set({0,1,1,1}, 0);
  c.set({1,0,0,0}, 0);
  c.set({1,0,0,1}, 0);
  c.set({1,0,1,0}, 0);
  c.set({1,0,1,1}, 0);
  c.set({1,1,0,0}, 0);
  c.set({1,1,0,1}, 0);
  c.set({1,1,1,0}, 0);
  c.set({1,1,1,1}, 0);

  DenseMMatrix ab({2,2,2});
  DenseMMatrix bc({2,2,2});
  DenseMMatrix ab_c({2,2,2});
  DenseMMatrix a_bc({2,2,2});
  Multiply(2, &a, &b, &ab);
  Multiply(2, &b, &c, &bc);
  Multiply(2, &ab, &c, &ab_c);
  Multiply(2, &a, &bc, &a_bc);

  bool are_equal = true;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        if (ab_c.get({i,j,k}) != a_bc.get({i,j,k})) {
          are_equal = false;
          // std::cout << i << ", " << j << ", " << k << std::endl;
          // std::cout << "  ab_c:" << ab_c.get({i,j,k}) << std::endl;
          // std::cout << "  a_bc:" << a_bc.get({i,j,k}) << std::endl;
        }
      }
    }
  }
  if (are_equal) {
    std::cerr << "ab_c and a_bc are equal but should not be" << std::endl;
    std::exit(1);
  }
}

void TestIdentityMultiplication() {
  std::cout << "TestIdentityMultiplication..." << std::endl;

  DenseMMatrix a({7,3,5});
  for (int i = 0; i < 7*3*5; i++) {
    a.set(i, i);
  }
  
  unique_ptr<MMatrixInterface> ident = Ident(2, a.shape());
  
  DenseMMatrix a_ident(a.shape());
  Multiply(a.shape().size(), &a, ident.get(), &a_ident);
  for (int i0 = 0; i0 < 7; i0++) {
    for (int i1 = 0; i1 < 3; i1++) {
      for (int i2 = 0; i2 < 5; i2++) {
        for (int i3 = 0; i3 < 7; i3++) {
          for (int i4 = 0; i4 < 3; i4++) {
            for (int i5 = 0; i5 < 5; i5++) {
              float expected = 0;
              if (i0 == i3 && i1 == i4 && i2 == i5) {
                expected = 1;
              }
              float actual = ident->get({i0, i1, i2, i3, i4, i5});
              if (actual != expected) {
                std::cout << "Index: ";
                for(int i : vector<int>{i0, i1, i2, i3, i4, i5}) {
                  std::cout << i << " ";
                }
                std::cout << "= " << actual << " expected " << expected;
                std::cout << std::endl;
              }
            }
          }
        }
      }
    }
  }
  if (!AreEqual(&a, &a_ident)) {
    std::cerr << "Error multiplying identity. Should be equal." << std::endl;
    std::exit(1);
  }
}

int main() {
  TestToFromVIndex();
  TestDenseMMatrixGetSet();
  TestMMatrixMultiplication();
  TestMMatrixMultiplicationAssociation();
  TestIdentityMultiplication();
  std::cout << "All tests pass." << std::endl;
  return 0;
}
