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

		igl::sort(unsorted, true, sorted, index_map);
		Eigen::Matrix3d uvw;

		// Set estimated N to eigenvector with the least eigenvalue
		estimated_N.row(i) = uvw_unsorted.col(index_map[0]);
	}
}

void find_support_radii_squared(
	const MatrixXd & P,
	const MatrixXi & kNN,
	VectorXd & support_radii)
{
	support_radii.resize(kNN.rows());

	for(int i=0; i<kNN.rows(); i++)
	{
		Vector3d curr_pos = P.row(i);

		double max = 0;
		for(int j=0; j<kNN.cols(); j++)
		{
			Vector3d other_pos = P.row(kNN(i, j));
			Vector3d diff = curr_pos - other_pos;
			double distance = diff.dot(diff);

			if(distance > max)
			{
				max = distance;
			}
		}

		support_radii(i) = max;
	}
}

void find_weights(
	const MatrixXd & P,
	const MatrixXd & N,
	const MatrixXi & kNN,
	MatrixXd & E,
	MatrixXd & W)
{
	E.resize(kNN.rows(), kNN.cols());
	W.resize(kNN.rows(), kNN.cols());

	// Find squared support radii
	VectorXd support_r;
	find_support_radii_squared(P, kNN, support_r);

	// Calculate potentials and weights for each edge
	for(int i=0; i<kNN.rows(); i++)
	{
		Vector3d pos_i = P.row(i);
		Vector3d n_i = N.row(i);
		double r_i = support_r(i);

		for(int j_index=0; j_index<kNN.cols(); j_index++)
		{
			int j = kNN(i, j_index);
			Vector3d pos_j = P.row(j);
			Vector3d n_j = N.row(j);
			double r_j = support_r(j);

			// find distances squared
			Vector3d diff = pos_i - pos_j;
			double distance = diff.dot(diff);

			// find max support radius squared
			double r_max = std::max(r_i, r_j);

			// find w
			double w = std::exp(-distance/r_max);

			// find flip criterion
			Vector3d e = diff.normalized();
			Vector3d n_i_new = n_i - e.dot(n_i) * e;

			double flip = n_i_new.dot(n_j);

			// find potential and weight
			double potential = flip * w;
			E(i, j_index) = potential;
			W(i, j_index) = std::abs(potential);
		}
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

	// (3.2) Graph Construction
	// Calculate potentials and weights for the edges
	// of the kNN graph
	MatrixXd E, W;
	find_weights(P, estimated_N, kNN, E, W);

	result_N = estimated_N;
}
