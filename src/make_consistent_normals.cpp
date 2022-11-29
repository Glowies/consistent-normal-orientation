#include "make_consistent_normals.h"
#include <igl/octree.h>
#include <igl/knn.h>

#include <iostream>

using namespace Eigen;

void estimate_normals(
  const MatrixXd & V,
	const MatrixXi & kNN,
	MatrixXd & estimated_N)
{
	int k = kNN.cols();
	estimated_N = MatrixXd::Zero(V.rows(), 3);

	for(int i=0; i<V.rows(); i++)
	{
		// - - - - - - - - - - - - - - - - - - - - - - - - - - - 
		// PCA code sourced from my submission for A7: Curvature
		// - - - - - - - - - - - - - - - - - - - - - - - - - - - 

		// Create relative positions matrix P
		Eigen::MatrixXd P(k, 3);
		for(int j=0; j<k; j++)
		{
			int adj_vertex = kNN(i, j);
			P.row(j) = V.row(adj_vertex) - V.row(i);
		}

		// Calculate covariance
		Eigen::Matrix3d Cov = P.transpose() * P;

		// Find principle components for plane
		auto solver = Eigen::EigenSolver<Eigen::Matrix3d>();
		auto solution = solver.compute(Cov, true);
		Eigen::Matrix3d uvw_unsorted = solution.eigenvectors().real();

		// Sort by eigenvalues
		Eigen::Vector3d eigenvalues = solution.eigenvalues().real();
		std::vector<double> unsorted {eigenvalues(0), eigenvalues(1), eigenvalues(2)};
		std::vector<double> sorted;
		std::vector<size_t> index_map;

		igl::sort(unsorted, false, sorted, index_map);
		Eigen::Matrix3d uvw;

		// Set estimated N to eigenvector with the least eigenvalue
		estimated_N.row(i) = uvw_unsorted.col(index_map[0]);
	}
}

void make_consistent_normals(
    const MatrixXd & P,
    const MatrixXd & N,
    MatrixXd & result_N)
{
	const int k = 16;
  result_N = MatrixXd::Zero(N.rows(), N.cols());

  // Build octree
	std::vector<std::vector<int>> O_PI;
	MatrixXi O_CH;
	MatrixXd O_CN;
	VectorXd O_W;
	igl::octree(P,O_PI,O_CH,O_CN,O_W);

	// Construct kNN graph
	MatrixXi kNN;
  igl::knn(P,k,O_PI,O_CH,O_CN,O_W,kNN);

	// (3.1) Normal Estimation
	// Estimate the normal for each point by applying PCA
	// on its k nearest neighbors
	MatrixXd estimated_N;
	estimate_normals(P, kNN, estimated_N);

	result_N = estimated_N;
}
