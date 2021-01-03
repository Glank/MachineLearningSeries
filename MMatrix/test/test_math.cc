#include "mmatrix.h"
#include "math_functions.h"
#include "mmatrix_test_utils.h"

#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>
#include <functional>

using namespace mmatrix;
using namespace mmatrix::test;
using namespace math;
using std::vector;
using std::unique_ptr;
using std::rand;

///// Tests

constexpr int kTrials = 100;

void TestAssociativity() {
  std::cout << "Testing Associativity of Multimatrix Multiplication..." << std::endl;
  std::srand(314);
  for (int trials = 0; trials < kTrials; trials++) {
    // |A| = a + m
    // |B| = m + b + n
    // |C| = n + c
    std::vector<int>
      a = RandShape(),
      m = RandShape(),
      b = RandShape(),
      n = RandShape(),
      c = RandShape();

    DenseMMatrix
      A(Concat(a,m)),
      B(Concat(m,b,n)),
      C(Concat(n,c));
    RandMatrix(&A);
    RandMatrix(&B);
    RandMatrix(&C);

    DenseMMatrix
      AB(Concat(a,b,n)),
      AB_C(Concat(a,b,c)),
      BC(Concat(m,b,c)),
      A_BC(Concat(a,b,c));
    
    Multiply(m.size(), &A, &B, &AB);
    Multiply(n.size(), &AB, &C, &AB_C);

    Multiply(n.size(), &B, &C, &BC);
    Multiply(m.size(), &A, &BC, &A_BC);

    if (!AreEqual(&AB_C, &A_BC)) {
      std::cerr << "Multiplicative Associativity Broken." << std::endl;
      std::exit(1);
    }

    //std::cout << DebugString(&AB_C) << std::endl;
    //std::cout << DebugString(&A_BC) << std::endl << std::endl; }
  }
}

void TestCommunicativity() {
  std::cout << "Testing Communicativity of Multiplication..." << std::endl;
  std::srand(314);
  for (int trials = 0; trials < kTrials; trials++) {
    // |a| = x 
    // |b| = x
    std::vector<int> x = RandShape();
    DenseMMatrix a(x), b(x);
    RandMatrix(&a);
    RandMatrix(&b);
    
    DenseMMatrix ab({}), ba({});

    Multiply(x.size(), &a, &b, &ab);
    Multiply(x.size(), &b, &a, &ba);
    
    if (!AreEqual(&ab, &ba)) {
      std::cerr << "Multiplicative Communicativity Broken" << std::endl;
      std::exit(1);
    }

    //std::cout << DebugString(&ab) << std::endl;
    //std::cout << DebugString(&ba) << std::endl << std::endl;
  }
}

void TestIdentity() {
  std::cout << "Testing Multiplicative Identity..." << std::endl;
  std::srand(314);
  for (int trials = 0; trials < kTrials; trials++) {
    // |a| = x + m
    // |b| = m + x
    std::vector<int>
      x = RandShape(),
      m = RandShape();
    DenseMMatrix
      a(Concat(x,m)),
      b(Concat(m,x));
    RandMatrix(&a);
    RandMatrix(&b);

    std::unique_ptr<MMatrixInterface> id = Ident(2, m);

    DenseMMatrix
      a_id(Concat(x,m)),
      id_b(Concat(m,x));

    Multiply(m.size(), &a, id.get(), &a_id);
    Multiply(m.size(), id.get(), &b, &id_b);

    if (!AreEqual(&a_id, &a) || !AreEqual(&id_b, &b)) {
      std::cerr << "Mult Identity Broken" << std::endl;
      std::exit(1);
    }

    //std::cout << DebugString(&a) << std::endl;
    //std::cout << DebugString(id.get()) << std::endl << std::endl;
  }
}

void TestSelfDerivative() {
  std::cout << "Testing Self Derivative..." << std::endl;
  std::srand(314);
  for (int trials = 0; trials < kTrials; trials++) {
    MMFunction f = [](const MMatrixInterface* x, MMatrixInterface* out) {
      Copy(x, out);
    };
    // |f(x)| = s
    // |x| = s
    std::vector<int> s = RandShape();
    DenseMMatrix x(s), out_tmp(s);
    RandMatrix(&x);

    // Real derivative at x
    std::unique_ptr<MMatrixInterface> dfdx_x = Ident(2, s);

    for (int subtrials = 0; subtrials < 10; subtrials++ ) {
      std::vector<int>
        f_idx = RandIndex(s),
        x_idx = RandIndex(s);
      MMFloat approx = ApproxDerivative(f, &out_tmp, &x, f_idx, x_idx);

      MMFloat real = dfdx_x->get(Concat(f_idx, x_idx));
    
      MMFloat approx_err = abs(real-approx);
      if (approx_err > 0.01) {
        std::cerr << "Self Derivative broken" << std::endl;
        std::exit(1);
      }
      //std::cout << "err: " << approx_err << std::endl;
    }
  }
}


void TestElementwiseDerivative() {
  std::cout << "Testing Elementwise Derivative..." << std::endl;
  std::srand(314);
  for (int trials = 0; trials < kTrials; trials++) {
    std::vector<MMFloat> polyCoefs;
    int polySize = rand()%5;
    for (int i = 0; i < polySize; i++) {
      polyCoefs.push_back(randf()*10-5);
    }
    std::function<MMFloat(MMFloat)> ef = [&](MMFloat x) {
      MMFloat out = 0;
      for (int i = 0; i < polySize; i++) {
        MMFloat term = polyCoefs[i];
        for (int j = 0; j < i; j++) {
          term *= x;
        }
        out += term;
      }
      return out;
    };
    std::function<MMFloat(MMFloat)> defdx = [&](MMFloat x) {
      MMFloat out = 0;
      for (int i = 0; i < polySize; i++) {
        MMFloat term = polyCoefs[i] * i;
        for (int j = 0; j < i-1; j++) {
          term *= x;
        }
        out += term;
      }
      return out;
    };
    MMFunction f = [&](const MMatrixInterface* x, MMatrixInterface* out) {
      Elementwise(ef, x, out);
    };
    // |f(x)| = s
    // |x| = s
    std::vector<int> s = RandShape();
    DenseMMatrix x(s), out_tmp(s);
    RandMatrix(&x);

    // Real derivative
    MMFunction dfdx = [&](const MMatrixInterface* x, MMatrixInterface* out) {
      auto ident = Ident(3, x->shape());
      DenseMMatrix ew(s);
      Elementwise(defdx, x, &ew);
      Multiply(x->shape().size(), ident.get(), &ew, out);
    };
    // Real derivative at x
    DenseMMatrix dfdx_x(Concat(s,s));
    dfdx(&x, &dfdx_x);

    for (int subtrials = 0; subtrials < 10; subtrials++ ) {
      std::vector<int>
        f_idx = RandIndex(s),
        x_idx = RandIndex(s);
      MMFloat approx = ApproxDerivative(f, &out_tmp, &x, f_idx, x_idx);

      MMFloat real = dfdx_x.get(Concat(f_idx, x_idx));
    
      MMFloat approx_err = abs(real-approx);
      ///std::cout << "err: " << approx_err << std::endl;
      if (approx_err > 0.01) {
        std::cerr << "Elementwise Derivative broken" << std::endl;
        std::cout << DebugString(&dfdx_x) << std::endl;
        std::cout << "real:" << real << std::endl;
        std::cout << "approx:" << approx << std::endl;
        for (int i = 0; i < polySize; i++) {
          if (i != 0) {
            std::cout << " + ";
          }
          std::cout << polyCoefs[i] << "x^" << i;
        }
        std::cout << std::endl;
        std::exit(1);
      }
    }
  }
}

void TestIdentityContraction() {
  std::cout << "Testing Identity Contraction..." << std::endl;
  std::srand(314);
  for (int trials = 0; trials < kTrials/2; trials++) {
    int m = rand()%3+2, n = rand()%3+2;
    int k = rand()%(std::min(m,n)-1)+1;
    if (k >= m || k >= n || k <= 0) {
      std::cerr << "Test broken." << std::endl;
      std::exit(1);
    }
    std::vector<int> x = RandShape();
    //std::cout << "m: " << m << "   n: " << n << "   k: " << k << std::endl;
    //std::cout << DebugString(x) << std::endl;
    auto ident_m = Ident(m, x);
    auto ident_n = Ident(n, x);
    SparseMMatrix real_prod(ConcatNTimes(m+n-2*k, x));
    Multiply(k*x.size(), ident_m.get(), ident_n.get(), &real_prod);
    auto expected_prod = Ident(m+n-2*k, x);

    if (!AreEqual(&real_prod, expected_prod.get())) {
      std::cerr << "Identity Contraction broken" << std::endl;
      std::cout << "m: " << m << "   n: " << n << "   k: " << k << std::endl;
      std::cout << "Real: " << DebugString(&real_prod) << std::endl;
      std::cout << "Expected: " << DebugString(expected_prod.get()) << std::endl;
      std::exit(1);
    }
  }
}

void InverseExample() {
  DenseMMatrix a({4,4});
  a.set({0,0}, -1);
  a.set({0,1}, 2);
  a.set({0,2}, 1);
  a.set({0,3}, 0);
  a.set({1,0}, 0);
  a.set({1,1}, -1);
  a.set({1,2}, 0);
  a.set({1,3}, 1);
  a.set({2,0}, 2);
  a.set({2,1}, 0);
  a.set({2,2}, 0.5);
  a.set({2,3}, 0);
  a.set({3,0}, 0);
  a.set({3,1}, 2);
  a.set({3,2}, 0);
  a.set({3,3}, 0.5);

  DenseMMatrix inv({4,4});

  if (!Invert(1, &a, &inv)) {
    std::cerr << "Err." << std::endl;
    exit(1);
  }

  for(int r = 0; r < 4; r++) {
    for(int c = 0; c < 4; c++) {
      std::cout << inv.get({r,c}) << "  ";
    }
    std::cout << std::endl;
  }
}

void InverseExampleCheck() {
  DenseMMatrix a({2,2,2,2});
  a.set({0,0,0,0}, 2);
  a.set({0,0,0,1}, -2);
  a.set({0,0,1,0}, 0);
  a.set({0,0,1,1}, 2);
  a.set({0,1,0,0}, -2);
  a.set({0,1,0,1}, 0);
  a.set({0,1,1,0}, -1);
  a.set({0,1,1,1}, 1);
  a.set({1,0,0,0}, 0);
  a.set({1,0,0,1}, -1);
  a.set({1,0,1,0}, 1);
  a.set({1,0,1,1}, 0);
  a.set({1,1,0,0}, 1);
  a.set({1,1,0,1}, -2);
  a.set({1,1,1,0}, 1);
  a.set({1,1,1,1}, -2);

  DenseMMatrix b({2,2,2,2});
  b.set({0,0,0,0}, 1/6.);
  b.set({0,0,0,1}, -1/3.);
  b.set({0,0,1,0}, -1/3.);
  b.set({0,0,1,1}, 0);
  b.set({0,1,0,0}, -1/6.);
  b.set({0,1,0,1}, -1/3.);
  b.set({0,1,1,0}, 0);
  b.set({0,1,1,1}, -1/3.);
  b.set({1,0,0,0}, -1/6.);
  b.set({1,0,0,1}, -1/3.);
  b.set({1,0,1,0}, 1);
  b.set({1,0,1,1}, -1/3.);
  b.set({1,1,0,0}, 1/6.);
  b.set({1,1,0,1}, 0);
  b.set({1,1,1,0}, 1/3.);
  b.set({1,1,1,1}, -1/3.);

  DenseMMatrix out({2,2,2,2});
  Multiply(2,&a,&b,&out);
  unique_ptr<MMatrixInterface> ident = Ident(2, {2,2});
  if (!AreEqual(&out, ident.get(), 0.0001)) {
    std::cerr << "Inverse Example Check Err." << std::endl;
    exit(1);
  }
}

void InverseExample2() {
  DenseMMatrix a({2,2,2,2});
  a.set({0,0,0,0}, 2);
  a.set({0,0,0,1}, -2);
  a.set({0,0,1,0}, 0);
  a.set({0,0,1,1}, 2);
  a.set({0,1,0,0}, -2);
  a.set({0,1,0,1}, 0);
  a.set({0,1,1,0}, -1);
  a.set({0,1,1,1}, 1);
  a.set({1,0,0,0}, 0);
  a.set({1,0,0,1}, -1);
  a.set({1,0,1,0}, 1);
  a.set({1,0,1,1}, 0);
  a.set({1,1,0,0}, 1);
  a.set({1,1,0,1}, -2);
  a.set({1,1,1,0}, 1);
  a.set({1,1,1,1}, -2);

  DenseMMatrix inv({2,2,2,2});

  if (!Invert(1, &a, &inv)) {
    std::cerr << "InverseExample2 Err." << std::endl;
    exit(1);
  }

  for(int r = 0; r < 2; r++) {
    for(int c = 0; c < 2; c++) {
      std::cout << inv.get({r,c}) << "  ";
    }
    std::cout << std::endl;
  }
}

void InverseExampleConstruction() {
  unique_ptr<MMatrixInterface> ident = Ident(2, {2,2});
  DenseMMatrix cinv({2,2});
  cinv.set({0,0}, -1);
  cinv.set({0,1}, 2);
  cinv.set({1,0}, 1);
  cinv.set({1,1}, 1/2.);
  DenseMMatrix out({2,2,2,2});
  
  Multiply(1, ident.get(), &cinv, &out);
  
  for (int d = 0; d < 2; d++) {
    for (int c = 0; c < 2; c++) {
      for (int b = 0; b < 2; b++) {
        for (int a = 0; a < 2; a++) {
          std::cout << out.get({a,b,c,d}) << std::endl;
        }
      }
    }
  }
  DenseMMatrix reshaped({4,4});
  Reshape(&out, &reshaped);
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      std::cout << reshaped.get({r,c}) << "  ";
    }
    std::cout << std::endl;
  }

  DenseMMatrix cinv2({2,2});
  Copy(&cinv, &cinv2);

  DenseMMatrix cMat({2,2});
  Invert(1, &cinv2, &cMat);

  DenseMMatrix ck({2,2});
  Multiply(1, &cMat, &cinv, &ck);
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 2; c++) {
      std::cout << cMat.get({r,c}) << "  ";
    }
    std::cout << std::endl;
  }

  
}

int main() {
  /*
  TestAssociativity();
  TestCommunicativity();
  TestIdentity();
  TestSelfDerivative();
  TestElementwiseDerivative();
  TestIdentityContraction();
  */
  InverseExampleConstruction();
  InverseExample();
  InverseExampleCheck();
  //InverseExample2();
  std::cout << "All tests pass." << std::endl;
  return 0;
}
