import numpy as np
def distance_matrix(vertices1, vertices2):
    v1_norm = np.sum(vertices1**2, axis=1)
    v2_norm = np.sum(vertices2**2, axis=1)
    v1_num = v1_norm.shape[0]
    v2_num = v2_norm.shape[0]
    dis_matrix = np.repeat(v1_norm, v2_num).reshape([-1, v2_num]) \
                 + np.repeat(np.expand_dims(v2_norm, axis=1).reshape([1,-1]), v1_num, axis=0) \
                 - 2*np.matmul(vertices1, np.transpose(vertices2))
    return dis_matrix

def local_maximum_filter(corner_pre_pro, corner_pre_points):
    dis = distance_matrix(corner_pre_points, corner_pre_points)

    # t = 0.006 0.003
    t = 0.003
    dist_t = dis
    dist_t[dist_t<=t] = 0
    dist_t[dist_t>t] = 1
    dist_t = 1 - dist_t
    prob_mask = np.repeat(np.expand_dims(corner_pre_pro, axis=0).reshape([1, -1]), len(corner_pre_pro), axis=0)
    dis_and_prob = dist_t * prob_mask

    max_idx = np.argmax(dis_and_prob, axis=1)
    idx_num = np.arange(len(corner_pre_pro))
    max_idx = max_idx - idx_num

    local_max_pro_idx = np.where(max_idx == 0)[0]
    return local_max_pro_idx

