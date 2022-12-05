#ifndef CLUSTER_EDGES_H
#define CLUSTER_EDGES_H
#include <Eigen/Dense>

void cluster_edges(
	const Eigen::MatrixXi & kNN,
	int iteration_count,
	Eigen::MatrixXi & edges,
	Eigen::VectorXd & E,
	Eigen::VectorXd & W,
	Eigen::VectorXi & collapse_target,
	Eigen::VectorXi & flipflag);

#endif