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

void TestSparseMultiplication() {
  std::cout << "TestSparseMultiplication..." << std::endl;

  SparseMMatrix a({2,2});
  a.set({0,0}, 2);
  a.set({1,1}, 7);
  
  SparseMMatrix b({2,3});
  b.set({0,1}, -2);
  b.set({0,2}, 3);
  b.set({1,1}, 5);
  b.set({1,2}, 6);

  SparseMMatrix out({2,3});
  Multiply(1, &a, &b, &out);

  SparseMMatrix expected({2,3});
  expected.set({0,1}, 2*-2);
  expected.set({0,2}, 2*3);
  expected.set({1,1}, 7*5);
  expected.set({1,2}, 7*6);

  if (!AreEqual(&out, &expected)) {
    std::cerr << "Error multiplying sparse matricies. Should be equal." << std::endl;
    std::exit(1);
  }
}

void TestSparseMultiplicationThorough() {
  std::cout << "TestSparseMultiplicationThorough..." << std::endl;

  std::srand(314);

  for (int trials = 0; trials < 100; trials++) {

    SparseMMatrix a_s ({5,5,5});
    DenseMMatrix a_d ({5,5,5});
    a_d.zero();
  
    SparseMMatrix b_s ({5,5,5});
    DenseMMatrix b_d ({5,5,5});
    b_d.zero();

    int a_vals = std::rand() % 125;
    for(int i = 0; i < a_vals; i++) {
      std::vector<int> index = {std::rand()%5, std::rand()%5, std::rand()%5};
      int val = std::rand()%11-5;
      a_s.set(index, val);
      a_d.set(index, val);
    }
    int b_vals = std::rand() % 125;
    for(int i = 0; i < b_vals; i++) {
      std::vector<int> index = {std::rand()%5, std::rand()%5, std::rand()%5};
      int val = std::rand()%11-5;
      b_s.set(index, val);
      b_d.set(index, val);
    }

    SparseMMatrix out_s({5,5});
    DenseMMatrix out_d({5,5});

    Multiply(2, &a_s, &b_s, &out_s);
    Multiply(2, &a_d, &b_d, &out_d);

    if (!AreEqual(&out_s, &out_d)) {
      std::cerr << "Error multiplying sparse matricies. Should be equal." << std::endl;
      std::exit(1);
    }
  }
}

void ManualTestSparseSwitch() {
  std::cout << "ManualTestSparseSwitch..." << std::endl;

  SparseMMatrix sparse({1});
  DenseMMatrix dense({1});
  DenseMMatrix out({1});

  Multiply(1, &dense, &dense, &out);
  Multiply(1, &sparse, &dense, &out);
  Multiply(1, &dense, &sparse, &out);
  Multiply(1, &sparse, &sparse, &out);
}

void TestAddTo() {
  std::cout << "TestAddTo..." << std::endl;

  SparseMMatrix a({2,3});
  a.set({0,0}, 2);
  a.set({1,1}, 7);
  
  SparseMMatrix b({2,3});
  b.set({0,1}, -2);
  b.set({0,2}, 3);
  b.set({1,1}, 5);
  b.set({1,2}, 6);

  SparseMMatrix out({2,3});
  AddTo(&a, &out);
  AddTo(&b, &out);

  SparseMMatrix expected({2,3});
  expected.set({0,0}, 2);
  expected.set({0,1}, -2);
  expected.set({0,2}, 3);
  expected.set({1,1}, 12);
  expected.set({1,2}, 6);

  if (!AreEqual(&out, &expected)) {
    std::cerr << "Error adding matricies. Should be equal." << std::endl;
    std::exit(1);
  }
}

void TestElementwise() {
  std::cout << "TestElementwise..." << std::endl;

  std::function<float(float)> f = [](float x) { return x*2; };

  SparseMMatrix m({2,1});
  m.set({0,0}, 1);
  m.set({1,0}, 3);

  SparseMMatrix out({2,1});
  
  Elementwise(f, &m, &out);

  SparseMMatrix expected({2,1});
  expected.set({0,0}, 2);
  expected.set({1,0}, 6);

  if (!AreEqual(&out, &expected)) {
    std::cerr << "Error applying elementwise op. Should be equal." << std::endl;
    std::exit(1);
  }
}

int main() {
  TestToFromVIndex();
  TestDenseMMatrixGetSet();
  TestMMatrixMultiplication();
  TestMMatrixMultiplicationAssociation();
  TestIdentityMultiplication();
  TestSparseMultiplication();
  TestSparseMultiplicationThorough();
  TestAddTo();
  TestElementwise();
  //ManualTestSparseSwitch();
  std::cout << "All tests pass." << std::endl;
  return 0;
}
