#include "cluster_edges.h"
#include <set>
#include <map>
#include <vector>
#include <iostream>

using namespace Eigen;
using namespace std;

void find_collapsible_edges(
	const MatrixXi & kNN,
	const MatrixXi & edges,
	const VectorXd & E,
	const VectorXd & W,
	VectorXd & max_edge_energy_per_vertex,
	VectorXi & max_weight_edge,
	VectorXi & collapse_target,
	VectorXi & edge_index_per_vertex
	)
{
	int n = kNN.rows();
	int edge_count = edges.rows();

	// find max energy per vertex 
	max_edge_energy_per_vertex = VectorXd::Zero(n);
	max_weight_edge = VectorXi::Ones(n) * -1;
	for(int edge_i=0; edge_i<edge_count; edge_i++)
	{
		Vector2i edge = edges.row(edge_i);
		int v1 = edges(edge_i, 0);
		int v2 = edges(edge_i, 1);
		auto edge_w = W(edge_i);

		if(edge_w > max_edge_energy_per_vertex(v1) )
		{
			max_edge_energy_per_vertex(v1) = edge_w;
			max_weight_edge(v1) = edge_i;
		}

		if(edge_w > max_edge_energy_per_vertex(v2) )
		{
			max_edge_energy_per_vertex(v2) = edge_w;
			max_weight_edge(v2) = edge_i;
		}
	}

	// find collapse targets
	collapse_target = VectorXi::Ones(n) * - 1;
	edge_index_per_vertex = VectorXi::Ones(n) * - 1;
	for(int edge_i=0; edge_i<edge_count; edge_i++)
	{
		Vector2i curr_edge = edges.row(edge_i);
		int v1 = edges(edge_i, 0);
		Vector2i edge1 = edges.row(max_weight_edge[v1]);

		int v2 = edges(edge_i, 1);
		Vector2i edge2 = edges.row(max_weight_edge[v2]);

		if (edge1 == edge2 && edge2 == curr_edge)
		{
			collapse_target(v2) = v1;
			edge_index_per_vertex(v2) = edge_i;
		}
		else if ((edge1 == curr_edge) != (edge2 == curr_edge))
		{
			if (edge1 == curr_edge)
			{
				collapse_target(v1) = v2;
				edge_index_per_vertex(v1) = edge_i;
			}
			if (edge2 == curr_edge)
			{
				collapse_target(v2) = v1;
				edge_index_per_vertex(v2) = edge_i;
			}
		}
	}
}

void find_collapse_chains(
	const MatrixXi & edges,
	const VectorXd & E,
	const VectorXd & W,
	VectorXi & max_weight_edge,
	VectorXi & collapse_target,
	VectorXi & edge_index_per_vertex
	)
{
	int edge_count = edges.rows();

	for(int edge_i=0; edge_i<edge_count; edge_i++)
	{
		Vector2i curr_edge = edges.row(edge_i);
		int v1 = edges(edge_i, 0);
		int v2 = edges(edge_i, 1);
		int edge1_i = max_weight_edge(v1);
		int edge2_i = max_weight_edge(v2);
		Vector2i edge1 = edges.row(edge1_i);
		Vector2i edge2 = edges.row(edge2_i);
		int col1 = collapse_target(v1);
		int col2 = collapse_target(v2);
		double w1 = W(edge1_i);
		double w2 = W(edge2_i);

		bool flag1 = (edge1 != curr_edge) && (edge2 != curr_edge);
		bool flag2 = (col1 != -1) && (col2 != -1);

		if(!(flag1 && flag2))
		{
			continue;
		}

		while(collapse_target(col1) != -1)
		{
			col1 = collapse_target(col1);
		}
		while(collapse_target(col2) != -1)
		{
			col2 = collapse_target(col2);
		}

		if(col1 != col2)
		{
			continue;
		}

		if(w1 < w2)
		{
			collapse_target(v1) = -1;
			edge_index_per_vertex(v1) = -1;
		}
		else
		{
			collapse_target(v2) = -1;
			edge_index_per_vertex(v2) = -1;
		}
	}
}

void handle_orientation_flips(
	const MatrixXi & edges,
	const VectorXi & edge_index_per_vertex,
	VectorXd & E,
	VectorXi & flipflag
)
{
	flipflag = VectorXi::Ones(edge_index_per_vertex.size());
	for(int i=0; i<edge_index_per_vertex.size(); i++)
	{
		int target_edge_i = edge_index_per_vertex(i);
		if(target_edge_i == -1)
		{
			continue;
		}

		Vector2i edge = edges.row(target_edge_i);

		if(E(target_edge_i) < 0)
		{
			E(target_edge_i) = -E(target_edge_i);
			flipflag(edge(0)) = -1;
			flipflag(edge(1)) = -1;
		}
	}
}

void collapse_edges(
	MatrixXi & edges,
	VectorXi & collapse_target
)
{
	int edge_count = edges.rows();

	for(int edge_i=0; edge_i<edge_count; edge_i++)
	{
		Vector2i curr_edge = edges.row(edge_i);

		int new1 = -1;
		int new2 = -1;

		// extend v1
		int temp = collapse_target(curr_edge(0));
		while (temp != -1)
		{
			new1 = temp;
			temp = collapse_target(temp);
		}

		// extend v2
		temp = collapse_target(curr_edge(1));
		while (temp != -1)
		{
			new2 = temp;
			temp = collapse_target(temp);
		}

		if (new1 == -1)
		{
			edges(edge_i, 0) = new1;
		}
		if (new2 == -1)
		{
			edges(edge_i, 1) = new2;
		}
		
		if (edges(edge_i, 0) > edges(edge_i, 1))
		{
			temp = edges(edge_i, 0);
			edges(edge_i, 0) = edges(edge_i, 1);
			edges(edge_i, 1) = edges(edge_i, 0);
		}
	}
}

void adjust_energies_and_weights(
	MatrixXi & edges,
	VectorXd & E,
	VectorXd & W
)
{
	std::map<std::pair<int, int>, int> edge_map = {};

	for(int i=0; i<edges.rows(); i++)
	{
		Vector2i edge = edges.row(i);
		std::pair<int, int> edge_pair {edge(0), edge(1)};

		// Just add to edge map if
		// edge has no duplicates yet
		if(edge_map.count(edge_pair) == 0)
		{
			edge_map[edge_pair] = i;
			continue;
		}

		// If edge has duplicates, calculate
		// cumulative energies and weights
		int prev_i = edge_map[edge_pair];
		E(i) = E(i) + E(prev_i);
		W(i) = std::max(W(i), W(prev_i));

		// delete the previous duplicate edge
		edges.row(prev_i) << -1, -1;

		// update edge map
		edge_map[edge_pair] = i;
	}
}

void clean_edges(
	MatrixXi & edges,
	VectorXd & E,
	VectorXd & W
)
{
	MatrixXi valid_edges;
	VectorXd valid_E, valid_W;
	valid_edges.resize(edges.rows(), 2);
	valid_E.resize(edges.rows());
	valid_W.resize(edges.rows());
	int valid_count = 0;

	std::set<std::pair<int, int>> edge_set = {};

	for(int i=0; i<edges.rows(); i++)
	{
		Vector2i edge = edges.row(i);
		std::pair<int, int> edge_pair {edge(0), edge(1)};

		// Skip edge if invalid or duplicate
		if(edge_set.count(edge_pair) > 0 || 
			edge(0) == edge(1) || 
			edge(0) < 0 || edge(1) < 1)
		{
			continue;
		}
		
		edge_set.insert(edge_pair);

		// Insert into valid lists
		valid_edges.row(valid_count) = edge;
		valid_E(valid_count) = E(i);
		valid_W(valid_count) = W(i);

		valid_count++;
	}

	// Resize and assign valid lists
	edges = valid_edges.topRows(valid_count);
	E = valid_E.topRows(valid_count);
	W = valid_W.topRows(valid_count);
}

void cluster_edges(
	const MatrixXi & kNN,
	MatrixXi & edges,
	VectorXd & E,
	VectorXd & W,
	VectorXi & collapse_target,
	VectorXi & flipflag
)
{
	// Pre-clean Edges
	// This gets rid of invalid and duplicate edges
	clean_edges(edges, E, W);

	VectorXd max_edge_energy_per_vertex;
	VectorXi max_weight_edge;
	VectorXi edge_index_per_vertex;

	// Keep collapsing edges while there are edges left
	while(edges.rows() > 0)
	{
		// Find Collapsible Edge
		find_collapsible_edges(kNN, edges, E, W, max_edge_energy_per_vertex, max_weight_edge, collapse_target, edge_index_per_vertex);
		
		// Find Collapse Chains
		find_collapse_chains(edges, E, W, max_weight_edge, collapse_target, edge_index_per_vertex);
		
		// Handle Orientation Flips
		handle_orientation_flips(edges, edge_index_per_vertex, E, flipflag);

		// Collapse Edges
		collapse_edges(edges, collapse_target);

		// Adjust Edge Energies and Weight
		adjust_energies_and_weights(edges, E, W);

		// Remove Collapsed Edges
		clean_edges(edges, E, W);
	}
}