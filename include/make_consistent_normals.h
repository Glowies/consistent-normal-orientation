#ifndef CONSISTENT_NORMALS_H
#define CONSISTENT_NORMALS_H
#include <Eigen/Dense>

void make_consistent_normals(
    const Eigen::MatrixXd & P,
    const Eigen::MatrixXd & N,
    Eigen::MatrixXd & result_N);

#endif