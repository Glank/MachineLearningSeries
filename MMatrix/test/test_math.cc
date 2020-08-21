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



int main() {
  TestAssociativity();
  TestCommunicativity();
  TestIdentity();
  TestSelfDerivative();
  TestElementwiseDerivative();
  TestIdentityContraction();
  std::cout << "All tests pass." << std::endl;
  return 0;
}
