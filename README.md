# Globally Consistent Normal Orientation of Raw Unorganized Point Clouds

This project implements the following paper:
https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13797

Given an unorganized ponit cloud, it estimates normals for each point by applying PCA on its nearest neighbors. After that, it creates a kNN graph using libigl and proceeds to collapse this graph using the greedy graph collapse algorithm outlined in the paper. It keeps track of the orientation flips performed during the graph collapse and reorients the estimated normals using this information.

## Compilation

```
mkdir build-release
cd build-release
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```

## Execution

Once built, you can execute the assignment from inside the `build-release/` directory using 

    ./consistent-normals [path to point cloud]

Some sample point cloud files can be found under `data/`

## Viewing the Results

When you run the built executable, you will be shown the original normals provided in the point cloud file. The following key presses will display the different results produced as the algorithm runs:

```
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
```

Pressing `c` will display the resulting normals produced by the algorithm.
If you are interesting in how the graph looks as the algorithm runs, you can press the numbers 1 through 9. On the first iteration the graph will be the full kNN graph and with each following iteration the number of edges will decrease until they hit 0.