#ifndef COVERING_H
#define COVERING_H

#include "mmatrix.h"

#include <memory>

namespace covering {

class CoveringMatrices {
 public:
  CoveringMatrices(std::unique_ptr<mmatrix::MMatrixInterface> fourier_basis);

  void update_weights(std::function<void(mmatrix::MMatrixInterface*)> update);

  // The covering and deriv by  weights
  const mmatrix::MMatrixInterface* sigmoid();
  const mmatrix::MMatrixInterface* sigmoid_deriv();

  // Regulation of weights minimizes the sum for all weights,
  //   (w^2-k^2)(w^2+(1-2p)k^2)
  // k >= 0
  // 0 <= p <= 1
  // where p closer to 1 increases the average sharpness of coverings
  // and larger k increases the maximum sharpness of coverings
  void set_reg_params(double k, double p);
  double regulation();
  const mmatrix::MMatrixInterface* regulation_deriv();

  // The primary error and deriv by weights
  // k >= 0
  // where larger k means fewer default coverings
  void set_primary_params(double k);
  double primary();
  const mmatrix::MMatrixInterface* primary_deriv();

  // The covering density and deriv by weights
  // 0 <= d <= 1
  // where larger d means a higher density
  void set_density_params(double d);
  double density();
  const mmatrix::MMatrixInterface* density_deriv();

  double continuity();
  const mmatrix::MMatrixInterface* continuity_deriv();

 private:

  // The base fourier transform that we will be attempting to cover.
  std::unique_ptr<mmatrix::MMatrixInterface> fourier_basis_;
  // The size of the fourier_basis_
  long size_;
  int cur_state_ = 1;
  mmatrix::MMatrixInterface* weights();
  std::unique_ptr<mmatrix::DenseMMatrix> weights_;

  int sigmoid_state_ = 0;
  std::unique_ptr<mmatrix::DenseMMatrix> sigmoid_;
  int sigmoid_deriv_state_ = 0;
  std::unique_ptr<mmatrix::SparseMMatrix> sigmoid_deriv_;

  double reg_k_ = 10;
  double reg_p_ = 0;
  int regulation_state_ = 0;
  double regulation_;
  int regulation_deriv_state_ = 0;
  std::unique_ptr<mmatrix::DenseMMatrix> regulation_deriv_;

  double prim_k_ = 10;
  int primary_state_ = 0;
  double primary_;
  int primary_deriv_state_ = 0;
  bool fourier_basis_regularized_update_needed_ = true;
  const mmatrix::MMatrixInterface* fourier_basis_regularized();
  std::unique_ptr<mmatrix::DenseMMatrix> fourier_basis_regularized_;
  std::unique_ptr<mmatrix::DenseMMatrix> primary_deriv_;

  double density_d_ = 0;
  int density_state_ = 0;
  double density_;
  int density_deriv_state_ = 0;
  std::unique_ptr<mmatrix::DenseMMatrix> density_deriv_;

  int continuity_state_ = 0;
  double continuity_;
  int continuity_deriv_state_ = 0;
  std::unique_ptr<mmatrix::DenseMMatrix> continuity_deriv_;
  
  // Ident^3(|weights|)
  const mmatrix::MMatrixInterface* ident3_w();
  std::unique_ptr<mmatrix::MMatrixInterface> ident3_w_;
  // A temporary vector in the shape |weights|
  mmatrix::MMatrixInterface* tmp_w();
  std::unique_ptr<mmatrix::DenseMMatrix> tmp_w_;
};

}  // covering

#endif // COVERING_H
