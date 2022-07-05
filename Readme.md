## Coarse-to-fine Pipeline for 3D Wireframe Reconstruction from Point Cloud

Point clouds captured by 3D scans are typically sparse, irregular, and noisy, resulting in 3D wireframes reconstructed by existing approaches often containing redundant edges or lacking proper edges. To tackle these issues, this paper proposes a coarse-to-fine pipeline for 3D wireframe reconstruction from point clouds. First, a learning-based module is dedicated to predicting the corner and edge points from the input point cloud, and each pair of corner points is linked together to generate an initial 3D wireframe. Second, a coarse pruning module is utilized to generate a coarse 3D wireframe, which is achieved by pruning observable redundant edges from the initial 3D wireframe based on the asymmetric Chamfer distance. Third, a refined pruning module is used to generate a refined 3D wireframe with correct topological structures, which can help prune redundant edges that are difficult to observe from the coarse 3D wireframe. Finally, a heuristic algorithm is exploited to fine-tune the refined 3D wireframe to ensure the final 3D wireframe preserves the characteristics of both vertical and parallel. The experimental results reveal that the proposed method significantly improves the performance of 3D wireframes reconstruction from point clouds on the large-scale ABC dataset and a challenging furniture dataset.  

### Our contributions are as follows:

- The proposed coarse-to-fine 3D wireframe reconstruction approach operates directly on 3D point clouds.
- Two pruning modules are used successively to prune redundant edges in the over-complete edge set to reconstruct a 3D wireframe with the correct topology.
- We propose to use edge points detected from point clouds as self-supervised labels for 3D wireframe reconstruction.
- We exploit the particle swarm optimization algorithm to preserve the characteristics of both vertical and parallel in 3D wireframes.
- Experiments on different datasets show that the proposed method achieves superior results to existing methods on the 3D wireframe reconstruction.

## Requirements:

An Tensorflow implementation of our work.

- Tensorflow 1.12.0
- CUDA 10.1
- Python 3.6

## Overview code directory:

${ROOT}/ \
 ├── models/ :contains model definition.\
 ├── tf_ops/ : contains function modules of the pointnet++ network need to be compiled. \
 ├── utils/ :contains some utility functions. \
 ├── geometry_optimization.py : PSO optimizes the geometry of the reconstructed 3D wireframe. \
 ├── test.py/ : testing scripts for corner and edge points classification\
 ├── train.py/ : training scripts for corner and edge points classification. \
 ├── README.md

### Implementation details:

Our proposed method is implemented based on tensorflow 1.12.0 and cuda 10.1. An Adam optimizer is applied to train our network with a learning rate initialized to 0.001, and then decayed to 0.7 of the current learning rate every 200000 steps. Our network is trained on the GPU for a number of 100 epochs.

## Our model：

**You can download the trained model to verify our results.**

- Our train_model \

  [[Our model]](https://drive.google.com/file/d/1oOXIdftdP97oCRoUj4l_wWXgJTSpwa2d/view?usp=sharing)

### Dataset:

- ABC Dataset: download from this link\

  https://github.com/wangxiaogang866/PIE-NET/tree/master/main/train_data

- Furniture Dataset:  download from this link\

  https://github.com/wangxiaogang866/PIE-NET/tree/master/main/test_data

### References:

- [1] Wang X, Xu Y, Xu K, et al. Pie-net: Parametric inference of point cloud edges[J]. Advances in neural information processing systems, 2020, 33: 20167-20178.

- [2] Yu L, Li X, Fu C W, et al. Ec-net: an edge-aware point set consolidation network[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 386-402.
