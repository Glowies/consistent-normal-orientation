#include "make_consistent_normals.h"

#include <iostream>

using namespace Eigen;

void make_consistent_normals(
    const MatrixXd & P,
    const MatrixXd & N,
    MatrixXd & result_N)
{
  result_N = MatrixXd::Ones(N.rows(), N.cols());
}
