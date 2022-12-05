#include "make_consistent_normals.h"
#include <igl/list_to_matrix.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>

void normal_to_color(
  const Eigen::MatrixXd & N,
  Eigen::MatrixXd & color
)
{
  color = Eigen::MatrixXd::Ones(N.rows(), N.cols()) * 0.5;
  color += N * 0.5;
}

void get_graph_points_and_colors(
  const Eigen::MatrixXd & P,
  const Eigen::MatrixXi & edges,
  const Eigen::VectorXd & E,
  Eigen::MatrixXd & P1,
  Eigen::MatrixXd & P2,
  Eigen::MatrixXd & color
)
{
  P1.resize(edges.rows(), 3);
  P2.resize(edges.rows(), 3);
  color.resize(edges.rows(), 3);

  if(edges.rows() < 1)
  {
    return;
  }

  Eigen::Vector3d orange = Eigen::Vector3d(1., .65, 0);
  Eigen::Vector3d blue = Eigen::Vector3d(0, 0, 1.);
  double max_E = E.maxCoeff();
  
  for(int i=0; i<edges.rows(); i++)
  {
    P1.row(i) = P.row(edges(i, 0));
    P2.row(i) = P.row(edges(i, 1));
    if(E(i) > 0)
    {
      color.row(i) = blue * E(i) / max_E;
    }
    else
    {
      color.row(i) = orange * -E(i) / max_E;
    }
  }
}

int main(int argc, char *argv[])
{
  // Load in points + normals from .pwn file
  Eigen::MatrixXd P,N;
  {
    Eigen::MatrixXd D;
    std::vector<std::vector<double> > vD;
    std::string line;
    std::fstream in;
    in.open(argc>1?argv[1]:"../data/hand.pwn");
    while(in)
    {
      std::getline(in, line);
      std::vector<double> row;
      std::stringstream stream_line(line);
      double value;
      while(stream_line >> value) row.push_back(value);
      if(!row.empty()) vD.push_back(row);
    }
    if(!igl::list_to_matrix(vD,D)) return EXIT_FAILURE;
    assert(D.cols() == 6 && "pwn file should have 6 columns");
    P = D.leftCols(3);
    N = D.rightCols(3);
  }

  // Make Normals Consistent
  Eigen::MatrixXd consistent_N;
  Eigen::MatrixXi edges;
  Eigen::VectorXd E;
  int iteration_count = 0;
  make_consistent_normals(P, N, iteration_count, edges, E, consistent_N);

  // Create a libigl Viewer object to toggle between point cloud and mesh
  igl::opengl::glfw::Viewer viewer;
  std::cout<<R"(
  P,p      view original normals
  C,c      view consistent normals
  0        view final graph (should have no edges)
  1        view graph on 1st iteration
  2        view graph on 2nd iteration
  3        view graph on 3rd iteration
  4        view graph on 4th iteration
  5        view graph on 5th iteration
  6        view graph on 6th iteration
  7        view graph on 7th iteration
  8        view graph on 8th iteration
  9        view graph on 9th iteration
)";
  const auto set_points = [&]()
  {
    Eigen::MatrixXd colors;
    normal_to_color(N, colors);
    viewer.data().clear();
    viewer.data().set_points(P,Eigen::RowVector3d(1,1,1));
    viewer.data().add_edges(P,(P+0.01*N).eval(),colors);
  };
  const auto set_consistent_points = [&]()
  {
    Eigen::MatrixXd colors;
    normal_to_color(consistent_N, colors);
    viewer.data().clear();
    viewer.data().set_points(P,Eigen::RowVector3d(1,1,1));
    viewer.data().add_edges(P,(P+0.01*consistent_N).eval(),colors);
  };
  const auto view_graph = [&]()
  {
    make_consistent_normals(P, N, iteration_count, edges, E, consistent_N);

    Eigen::MatrixXd colors;
    normal_to_color(consistent_N, colors);
    viewer.data().clear();
    viewer.data().set_points(P,Eigen::RowVector3d(1,1,1));
    viewer.data().add_edges(P,(P+0.01*consistent_N).eval(),colors);

    // add graph edges
    Eigen::MatrixXd P1, P2, graph_colors;
    get_graph_points_and_colors(P, edges, E, P1, P2, graph_colors);
    viewer.data().add_edges(P1, P2, graph_colors);
  };
  set_points();
  viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer&, unsigned int key,int)
  {
    switch(key)
    {
      case 'P':
      case 'p':
        set_points();
        return true;
      case 'C':
      case 'c':
        set_consistent_points();
        return true;
      case '0':
        iteration_count = 0;
        view_graph();
        return true;
      case '1':
        iteration_count = 1;
        view_graph();
        return true;
      case '2':
        iteration_count = 2;
        view_graph();
        return true;
      case '3':
        iteration_count = 3;
        view_graph();
        return true;
      case '4':
        iteration_count = 4;
        view_graph();
        return true;
      case '5':
        iteration_count = 5;
        view_graph();
        return true;
      case '6':
        iteration_count = 6;
        view_graph();
        return true;
      case '7':
        iteration_count = 7;
        view_graph();
        return true;
      case '8':
        iteration_count = 8;
        view_graph();
        return true;
      case '9':
        iteration_count = 9;
        view_graph();
        return true;
    }
    return false;
  };
  viewer.data().point_size = 2;
  viewer.launch();

  return EXIT_SUCCESS;
}

