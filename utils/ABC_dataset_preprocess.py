import os
import numpy as np
import glob

# np.random.seed(1)

def resample_pcd(pcd ,n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.arange(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

class ABCDataset():
    def __init__(self, root='train_data', num_input_points=2048, train=True):
        self.num_input_points = num_input_points
        self.train = train
        self.root = root
        if self.train:
            self.datapath = glob.glob(os.path.join(self.root, 'ABC_train*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'ABC_test*.npz'))

    def __getitem__(self, index):
        fn = self.datapath[index]
        data = np.load(fn)
        pointcloud = data['pointcloud']
        edge_points = data['labels_edge_p']
        corner_points = data['labels_corner_p']
        if self.train:
            pointcloud = resample_pcd(pointcloud, self.num_input_points)
            n = pointcloud.shape[0]
            sample_idx = np.random.choice(n, self.num_input_points, replace=False)
            input_pointcloud = np.copy(pointcloud[sample_idx])
            label_edge_points = np.copy(edge_points[sample_idx])
            label_corner_points = np.copy(corner_points[sample_idx])
        else:
            pointcloud = resample_pcd(pointcloud, self.num_input_points)
            input_pointcloud = pointcloud
            label_edge_points = edge_points
            label_corner_points = corner_points

        return input_pointcloud, label_edge_points, label_corner_points

    def __len__(self):
        return len(self.datapath)