#pragma once

#include <string>
// #include <Eigen/Dense>

namespace param_id {
std::string test();

// inline Eigen::MatrixXd fdbk(
//   const Eigen::MatrixXd& A,
//   const Eigen::VectorXd& b,
//   const Eigen::MatrixXd& pinv,
//   const Eigen::VectorXd& x0 = Eigen::VectorXd(),
//   int max_iter = 10000,
//   double tol = 1e-6
// ) {

//   Eigen::MatrixXd A_tilde = pinv * A;

//   // Initialize the variables.
//   int dim = A.cols();
//   Eigen::VectorXd x;

//   if (x0.size() == 0)
//     x = Eigen::VectorXd::Zero(dim);
//   else
//     x = x0;

//   Eigen::VectorXd b_tilde = pinv * b;

//   // Iterate until the stopping criterion is met.
//   int iter = 0;
//   for (; iter < max_iter; ++iter) {
//     Eigen::VectorXd residual = b_tilde - A_tilde * x;

//     // Compute epsilon_k
//     double residual_norm = residual.squaredNorm();
    
//     Eigen::VectorXd matrix_row_norm = A_tilde.rowwise().squaredNorm();

//     if (matrix_row_norm.isZero()) {
//       // std::cout << "Matrix row norm is zero" << std::endl;
//       break;
//     }

//     Eigen::VectorXd res_over_row = residual.array().abs2() / matrix_row_norm.array();
//     double frob_norm = A_tilde.squaredNorm();
//     double epsilon_k = 0.5 * ((1 / residual_norm) * res_over_row.maxCoeff() + (1 / frob_norm));

//     // Determine the index set of positive integers
//     Eigen::VectorXd criterion = epsilon_k * residual_norm * matrix_row_norm;

//     std::vector<int> idxs;
//     Eigen::VectorXd tau_k = (residual.array().abs2() >= criterion.array()).cast<double>();

//     for (int i = 0; i < tau_k.size(); i++) {
//       if (tau_k(i) == 1.0) {
//         idxs.push_back(i);
//       }
//     }

//     Eigen::RowVectorXd tau_residuals = residual(idxs);
//     Eigen::MatrixXd id = Eigen::MatrixXd::Identity(dim, dim);        
//     Eigen::MatrixXd tau_id = id(Eigen::all, idxs);       
//     Eigen::VectorXd eta_k = tau_residuals * tau_id.transpose();
    
//     // Update x using the FDBK formula
//     Eigen::VectorXd step_size = ((eta_k.dot(residual)) / (A_tilde.transpose() * eta_k).squaredNorm()) * A_tilde.transpose() * eta_k;

//     x += step_size;

//     if ((b_tilde - A_tilde * x).norm() < tol) {
//       break;
//     }
//   }

//   // std::cout << "iterations: " << iter << "\n\n";

//   return x;
// }
}
