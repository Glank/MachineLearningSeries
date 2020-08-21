#include "covering.h"

#include "math_functions.h"

#include <iostream>
#include <algorithm>

using namespace mmatrix;

namespace covering {

CoveringMatrices::CoveringMatrices(std::unique_ptr<MMatrixInterface> fourier_basis)
  : fourier_basis_(std::move(fourier_basis)) {
  size_ = 1;
  for (int s : fourier_basis_->shape()) {
    size_ *= s;
  }
}

mmatrix::MMatrixInterface* CoveringMatrices::weights() {
  if (weights_ == nullptr) {
    weights_ = std::unique_ptr<DenseMMatrix>(new DenseMMatrix(fourier_basis_->shape()));
  }
  return weights_.get();
}

const mmatrix::MMatrixInterface* CoveringMatrices::ident3_w() {
  if (ident3_w_ == nullptr) {
    ident3_w_ = Ident(3, weights()->shape());
  }
  return ident3_w_.get();
}

mmatrix::MMatrixInterface* CoveringMatrices::tmp_w() {
  if (tmp_w_ == nullptr) {
    tmp_w_ = std::unique_ptr<DenseMMatrix>(new DenseMMatrix(fourier_basis_->shape()));
  }
  return tmp_w_.get();
}

void CoveringMatrices::update_weights(std::function<void(MMatrixInterface*)> update) {
  update(weights());
  cur_state_++;
}

const MMatrixInterface* CoveringMatrices::sigmoid() {
  if (sigmoid_state_ < cur_state_) {
    if (sigmoid_ == nullptr) {
      sigmoid_ = std::unique_ptr<DenseMMatrix>(new DenseMMatrix(weights()->shape()));
    }

    Elementwise([](MMFloat x){ return math::sigmoid(x); }, weights(), sigmoid_.get());

    sigmoid_state_ = cur_state_;
  }
  return sigmoid_.get();
}

const MMatrixInterface* CoveringMatrices::sigmoid_deriv() {
  if (sigmoid_deriv_state_ < cur_state_) {
    if (sigmoid_deriv_ == nullptr) {
      sigmoid_deriv_ = std::unique_ptr<SparseMMatrix>(new SparseMMatrix(Concat(weights()->shape(), weights()->shape())));
    }

    Elementwise([](MMFloat x){ return math::sigmoid_deriv(x); }, weights(), tmp_w());
    Multiply(2, ident3_w(), tmp_w(), sigmoid_deriv_.get());

    sigmoid_deriv_state_ = cur_state_;
  }
  return sigmoid_deriv_.get();
}

void CoveringMatrices::set_reg_params(double k, double p) {
  regulation_state_ = 0;
  regulation_deriv_state_ = 0;
  reg_k_ = k;
  reg_p_ = p;
}
double CoveringMatrices::regulation() {
  if (regulation_state_ < cur_state_) {
    Elementwise([&](MMFloat w){
      double w2 = w*w;
      double w4 = w2*w2;
      double k2 = reg_k_*reg_k_;
      double k4 = k2*k2;
      return (w4-2*reg_p_*k2*w2+k4*(2*reg_p_-1))/4;
    }, weights(), tmp_w());
    regulation_ = Sum(tmp_w());
    
    regulation_state_ = cur_state_;
  }
  return regulation_;
}
const mmatrix::MMatrixInterface* CoveringMatrices::regulation_deriv() {
  if (regulation_deriv_state_ < cur_state_) {
    if (regulation_deriv_ == nullptr) {
      regulation_deriv_ = std::unique_ptr<DenseMMatrix>(new DenseMMatrix(weights()->shape()));
    }
    
    Elementwise([&](MMFloat w){
      double w3 = w*w*w;
      return w3-reg_p_*reg_k_*reg_k_*w;
    }, weights(), regulation_deriv_.get());

    regulation_deriv_state_ = cur_state_;
  }
  return regulation_deriv_.get();
}

void CoveringMatrices::set_primary_params(double k) {
  primary_state_ = 0;
  primary_deriv_state_ = 0;
  prim_k_ = k;
  fourier_basis_regularized_update_needed_ = true;
}

const mmatrix::MMatrixInterface* CoveringMatrices::fourier_basis_regularized() {
  if (fourier_basis_regularized_ == nullptr) {
    fourier_basis_regularized_ = std::unique_ptr<DenseMMatrix>(new DenseMMatrix(weights()->shape()));
  }
  if (fourier_basis_regularized_update_needed_) {
    Elementwise([&](MMFloat x) {
      return math::sigmoid(std::log(std::max(x,1.0))-prim_k_);
    }, fourier_basis_.get(), fourier_basis_regularized_.get());
    fourier_basis_regularized_update_needed_ = false;
  }
  return fourier_basis_regularized_.get();
}

double CoveringMatrices::primary() {
  if (primary_state_ < cur_state_) {
    Copy(sigmoid(), tmp_w());
    SubFrom(fourier_basis_regularized(), tmp_w());
    primary_ = 0.5*SquaredSum(tmp_w());

    primary_state_ = cur_state_;
  }
  return primary_;
}

const mmatrix::MMatrixInterface* CoveringMatrices::primary_deriv() {
  if (primary_deriv_state_ < cur_state_) {
    if (primary_deriv_ == nullptr) {
      primary_deriv_ = std::unique_ptr<DenseMMatrix>(new DenseMMatrix(weights()->shape()));
    }
    sigmoid_deriv();
    Copy(sigmoid(), tmp_w());
    SubFrom(fourier_basis_regularized(), tmp_w());
    Multiply(2, tmp_w(), sigmoid_deriv(), primary_deriv_.get()); 

    primary_deriv_state_ = cur_state_;
  }

  return primary_deriv_.get();
}

void CoveringMatrices::set_density_params(double d) {
  density_d_ = d;
  density_state_ = 0;
  density_deriv_state_ = 0;
}

double CoveringMatrices::density() {
  if (density_state_ < cur_state_) {
    density_ = density_d_*size_ - Sum(sigmoid());
    density_ = density_ * density_ * 0.5;

    density_state_ = cur_state_;
  }
  return density_;
}

const mmatrix::MMatrixInterface* CoveringMatrices::density_deriv() {
  if (density_deriv_state_ < cur_state_) {
    if (density_deriv_ == nullptr) {
      density_deriv_ = std::unique_ptr<DenseMMatrix>(new DenseMMatrix(weights()->shape()));
    }

    double d = Sum(sigmoid())-density_d_*size_;
    Elementwise([&](MMFloat x){
      return d*math::sigmoid_deriv(x);
    }, weights(), density_deriv_.get());

    density_deriv_state_ = cur_state_;
  }

  return density_deriv_.get();
}

double CoveringMatrices::continuity() {
  if (continuity_state_ < cur_state_) {
    long frequencies = weights()->shape()[0];
    long samples = weights()->shape()[1];
    continuity_ = 0;
    for (long f = 0; f < frequencies; f++) {
      for (long n = 1; n < samples; n++) {
        double term = sigmoid()->get(f+frequencies*n)-sigmoid()->get(f+frequencies*(n-1));
        continuity_ += term*term;
      }
    }
    continuity_ *= 0.5;

    continuity_state_ = cur_state_;
  }
  return continuity_;
}

const mmatrix::MMatrixInterface* CoveringMatrices::continuity_deriv() {
  if (continuity_deriv_state_ < cur_state_) {
    if (continuity_deriv_ == nullptr) {
      continuity_deriv_ = std::unique_ptr<DenseMMatrix>(new DenseMMatrix(weights()->shape()));
    }

    sigmoid(); // ensure pre-calculated
    sigmoid_deriv(); // ensure pre-calculated
    long frequencies = weights()->shape()[0];
    long samples = weights()->shape()[1];
    for (long f = 0; f < frequencies; f++) {
      {
        double term = sigmoid()->get(f+frequencies*0);
        term -= sigmoid()->get(f+frequencies*1);
        tmp_w()->set(f+frequencies*0, term);
      }
      for (long n = 1; n < samples-1; n++) {
        double term = 2*sigmoid()->get(f+frequencies*n);
        term -= sigmoid()->get(f+frequencies*(n-1));
        term -= sigmoid()->get(f+frequencies*(n+1));
        tmp_w()->set(f+frequencies*n, term);
      }
      {
        double term = sigmoid()->get(f+frequencies*(samples-1));
        term -= sigmoid()->get(f+frequencies*(samples-2));
        tmp_w()->set(f+frequencies*(samples-1), term);
      }
    }
    Multiply(2, tmp_w(), sigmoid_deriv(), continuity_deriv_.get());

    continuity_deriv_state_ = cur_state_;
  }
  return continuity_deriv_.get();
}

}  // covering
