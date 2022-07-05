import numpy as np

def compute_chamfer_distance_np(p1, p2):
    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = np.sum(diff*diff, axis=3)
    dist1 = dist
    dist2 = np.transpose(dist, (0, 2, 1))

    dist_min1 = np.min(dist1, axis=2)
    dist_min2 = np.min(dist2, axis=2)

    return  dist_min1, dist_min2

def ComputeCD_np(CornerPair, WF_Shape):
    dist1, dist2 = compute_chamfer_distance_np(CornerPair, WF_Shape)
    distance = np.sum(dist1)
    return distance