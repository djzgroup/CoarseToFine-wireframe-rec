import time

import numpy as np
import os
from glob import glob
import argparse
from utils.local_maximum_filter import local_maximum_filter
from utils.chamfer_distance import ComputeCD_np
from utils.calculate_cosine import cosine_angle
from models.wf_pso import PSO
from tqdm import tqdm
import open3d as o3d
curr_dir = os.path.dirname(os.path.realpath(__file__))

def gen_datapoint(pointcloud, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in range(pointcloud.shape[0]):
            f.write((str(pointcloud[i, 0]) + '\t' + str(pointcloud[i, 1]) + '\t' + str(pointcloud[i, 2]) + '\n'))

def distance_optimization(corner_pairs, edge_points, args):
    distance_list = []
    eN, eC = edge_points.shape
    line_density = args.line_num_point
    for index, pair in enumerate(corner_pairs):
        line_vector = pair[0] - pair[1]
        scale = np.linspace(0, 1, line_density)
        line_points = []
        for ln in scale:
            line_point = pair[1] + ln * line_vector
            line_points.append(line_point)
        line_points = np.array(line_points)
        lN, lC = line_points.shape
        line_points = np.reshape(line_points, [1, lN, lC])
        distance = ComputeCD_np(line_points, np.reshape(edge_points, [1, eN, eC]))
        distance_list.append(distance)

    distance_list = np.array(distance_list)
    threshold_d = np.min(distance_list) * 10

    existent_pairs = []
    existent_lines_distance = []
    for index, d in enumerate(distance_list):
        if d < threshold_d:
            existent_pairs.append(corner_pairs[index])
            existent_lines_distance.append(d)
    return existent_pairs, existent_lines_distance


def angle_optimization(existent_pairs, existent_lines_distance, args):
    existent_lines_distance = np.array(existent_lines_distance)
    error_pairs_indices = []
    for index1, pair_s in enumerate(existent_pairs):
        for index2, pair_t in enumerate(existent_pairs):
            if index2 == index1:
                continue
            if np.array_equal(pair_t[0], pair_s[0]):
                vector_1 = pair_s[1] - pair_s[0]
                vector_2 = pair_t[1] - pair_t[0]
                cosine_value = cosine_angle(vector_1, vector_2)
                angle_hu = np.arccos(cosine_value)
                angle = angle_hu * 180 / np.pi
                distance1 = existent_lines_distance[index1]
                distance2 = existent_lines_distance[index2]
                if angle < args.angle_threshold:
                    if distance1 < distance2:
                        error_pairs_indices.append(index2)
                    else:
                        error_pairs_indices.append(index1)
    right_pairs = []
    right_distance = []
    for index, pair in enumerate(existent_pairs):
        if index not in error_pairs_indices:
            right_pairs.append(pair)
            right_distance.append(existent_lines_distance[index])

    final_pairs = []
    for ind, i in enumerate(right_pairs):
        flag = 0
        for j in final_pairs:
            if np.array_equal(j[0], i[1]) and np.array_equal(j[1], i[0]):
                flag = 1
                break
        if flag == 0:
            final_pairs.append(right_pairs[ind])
    return final_pairs

def pso_optimization(corner_pairs, edge_points, args):
    point = []
    for i in range(len(corner_pairs)):
        for j in range(len(corner_pairs[0])):
            point.append([corner_pairs[i][j][0], corner_pairs[i][j][1], corner_pairs[i][j][2]])

    map = {}
    for i in range(len(point)):
        map[i] = [point[i], 0]
    same = []
    for i in range(len(map) - 1):
        if map[i][1] == 1:
            continue
        else:
            s = [i]
            map[i][1] = 1
        for j in range(i + 1, len(map)):
            if map[j][0] == map[i][0]:
                s.append(j)
                map[j][1] = 1
        same.append(s)
    ml = 100
    v = 0
    point = np.array(point)
    for i in range(len(corner_pairs)):
        vector = point[2*i+1] - point[2*i]
        v = np.linalg.norm(vector)
        if v < ml:
            ml = v
    ml = v / args.scale
    pso = PSO(point, same, edge_points, ml, args)
    vertexes = pso.evolve()
    return vertexes

def line_to_obj(line_pred, save_to_path):
    with open(save_to_path, 'w') as f:
        for v in range(line_pred.shape[0]):
            f.write(f'v {line_pred[v][0]} {line_pred[v][1]} {line_pred[v][2]}\n')
        for l in range(line_pred.shape[0] // 2):
            f.write(f'l {2*l + 1} {2*l + 2}\n')

def evaluate(gt_corners, predicted_corners, line_th):
    th = line_th
    point_num_in_line = 30
    gt_points = []
    predicted_points = []
    for i in range(gt_corners.shape[0] // 2):
        e1_coord, e2_coord = gt_corners[i], gt_corners[i+1]
        for inter_point in range(1, point_num_in_line+1):
            inter_point_coord = (float(inter_point)/(point_num_in_line+1)*e1_coord + (1-float(inter_point)/(point_num_in_line+1))*e2_coord)
            gt_points.append(inter_point_coord)
    gt_points = np.array(gt_points)
    for i in range(predicted_corners.shape[0] // 2):
        e1_coord, e2_coord = predicted_corners[i], predicted_corners[i+1]
        for inter_point in range(1, point_num_in_line+1):
            inter_point_coord = (float(inter_point)/(point_num_in_line+1)*e1_coord + (1-float(inter_point)/(point_num_in_line+1))*e2_coord)
            predicted_points.append(inter_point_coord)
    predicted_points = np.array(predicted_points)
    gt = o3d.geometry.PointCloud()
    gt.points = o3d.utility.Vector3dVector(gt_points)
    pr = o3d.geometry.PointCloud()
    pr.points = o3d.utility.Vector3dVector(predicted_points)
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)
    recall = float(sum(d < th for d in d2)) / float(len(d2))
    precision = float(sum(d < th for d in d1)) / float(len(d1))
    if recall + precision > 0:
        fscore = 2 * recall * precision / (recall + precision)
    else:
        fscore = 0
    return  recall, precision, fscore

def read_obj_file(fname):
    vertices = []
    line_indices = []
    try:
        f = open(fname)
        for line in f:
            if line[0] == "v":
                strs = line.split()
                v0 = float(strs[1])
                v1 = float(strs[2])
                v2 = float(strs[3])
                vertex = [v0, v1, v2]
                vertices.append(vertex)

            elif line[0] == "l":
                strs = line.split()
                e1_index = int(strs[1])-1
                e2_index = int(strs[2])-1
                line_index = [e1_index, e2_index]
                line_indices.append(line_index)

        f.close()
    except IOError:
        print(".obj file not found.")

    vertices = np.array(vertices)
    line_indices = np.array(line_indices)
    lines = []
    for line_index in line_indices:
        end_point_1 = vertices[line_index[0]]
        end_point_2 = vertices[line_index[1]]
        lines.append([end_point_1, end_point_2])
    lines = np.array(lines)
    return lines

def main(args):
    files = glob(os.path.join(curr_dir, f'test_result/ECR-Net/test_PC2WF_datasets/*npz'))
    save_to_dir = os.path.join(curr_dir, f'visualize_line/ECR-Net/test_visual_PC2WF_datasets')
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)
    WF_fsocre = 0
    F_Recall = 0
    F_Precision = 0
    pre_edge_recall = 0
    pre_edge_precision = 0
    pre_corner_recall = 0
    pre_corner_precision = 0
    num_data = 0
    for file in tqdm(files):
        print('\n'+file)
        num_data += 1
        file_name = file.split('.')[0] + '.obj'
        gt_wf = os.path.join(curr_dir, file_name.replace('test_visual', 'WF_GT'))
        gt_corner_pairs = read_obj_file(gt_wf)
        data = np.load(file)
        pointcloud = data['pointcloud']
        gt_edge_labels = data['edge_label']
        gt_edge_labels_indices = np.where(gt_edge_labels == 1)[0]
        gt_edge_points = pointcloud[gt_edge_labels_indices]
        mask_edge = np.sum(gt_edge_points)
        gt_corner_labels = data['corner_label']
        gt_corner_labels_indices = np.where(gt_corner_labels == 1)[0]
        gt_corner_points = pointcloud[gt_corner_labels_indices]
        mask_corner = np.sum(gt_corner_points)


        edge_points_pre = data['pred_labels_edge_p']
        edge_points_pre = np.exp(edge_points_pre)
        sum_edge_pre = np.sum(edge_points_pre, axis=1)
        edge_points_pre = edge_points_pre / np.repeat(sum_edge_pre, 2).reshape([-1, 2])
        edge_points_label_pre = np.where(edge_points_pre[:, 1] > args.edge_points_threshold)[0]
        if len(edge_points_label_pre) == 0:
            continue
        edge_points_label = np.zeros([edge_points_pre.shape[0], 1])
        edge_points_label[edge_points_label_pre] = 1
        edge_pre_points = pointcloud[edge_points_label_pre, :]
        pre_edge_labels = np.int16(np.squeeze(edge_points_label))
        pre_edge_recall += np.sum(pre_edge_labels & gt_edge_points) / mask_edge
        pre_edge_precision += np.sum(pre_edge_labels & gt_edge_points) / np.sum(pre_edge_labels)

        corner_points_pre = data['pred_labels_corner_p']
        corner_points_pre = np.exp(corner_points_pre)
        sum_corner_pre = np.sum(corner_points_pre, axis=1)
        corner_points_pre = corner_points_pre / np.repeat(sum_corner_pre, 2).reshape([-1, 2])
        corner_points_label_pre = np.where(corner_points_pre[:, 1] > args.corner_points_threshold)[0]
        if len(corner_points_label_pre) == 0:
            continue
        corner_pre_points = pointcloud[corner_points_label_pre, :]
        corner_pre_pro = corner_points_pre[corner_points_label_pre, 1]
        local_max_pro_idx = local_maximum_filter(corner_pre_pro, corner_pre_points)
        global_corner_idx = corner_points_label_pre[local_max_pro_idx]
        corner_label_filter = np.zeros([corner_points_pre.shape[0], 1])
        corner_label_filter[global_corner_idx] = 1
        corner_points_label = corner_label_filter
        corner_int = np.int16(corner_points_label) * 2
        edge_int = np.int16(edge_points_label)
        corner_edge_intersection = np.squeeze(corner_int - edge_int)
        corner_pre_points = pointcloud[corner_edge_intersection==1, :]
        pre_corner_labels = []
        for element in corner_edge_intersection:
            if element == -1:
                element = 0
            pre_corner_labels.append(element)
        pre_corner_labels = np.array(pre_corner_labels)
        pre_corner_recall += np.sum(pre_corner_labels & gt_corner_points) / mask_corner
        pre_corner_precision += np.sum(pre_corner_labels & gt_corner_points) / np.sum(pre_corner_labels)
        corner_pairs = []
        for corner in corner_pre_points:
            for point in corner_pre_points:
                if np.array_equal(point, corner):
                    continue
                else:
                    corner_pairs.append([corner, point])
        existent_pairs, existent_lines_distance = distance_optimization(corner_pairs, edge_pre_points, args)
        final_corner_pairs = angle_optimization(existent_pairs, existent_lines_distance, args)
        pre_corner_points = []
        for i in range(len(final_corner_pairs)):
            for j in range(len(final_corner_pairs[0])):
                pre_corner_points.append([final_corner_pairs[i][j][0], final_corner_pairs[i][j][1], final_corner_pairs[i][j][2]])
        pre_corner_points = np.array(pre_corner_points)
        save_to_path = os.path.join(save_to_dir, file.split('/')[-1].replace('.npz', '_non_pso_pred.obj'))
        line_to_obj(pre_corner_points, save_to_path)
        if len(final_corner_pairs) == 1:
            continue
        pre_corner_points = pso_optimization(final_corner_pairs, edge_pre_points, args)
        save_to_path = os.path.join(save_to_dir, file.split('/')[-1].replace('.npz', '_pred.obj'))
        line_to_obj(pre_corner_points, save_to_path)

        ##########
        gt_corner_points = []
        for i in range(len(gt_corner_pairs)):
            for j in range(len(gt_corner_pairs[0])):
                gt_corner_points.append([gt_corner_pairs[i][j][0], gt_corner_pairs[i][j][1], gt_corner_pairs[i][j][2]])
        gt_corner_points = np.array(gt_corner_points)

        recall, precision, fscore = evaluate(gt_corner_points, pre_corner_points, args.line_threshold)
        F_Recall += recall
        F_Precision += precision
        WF_fsocre += fscore
    print('Edge Points detection Recall: %f' % (pre_edge_recall / num_data))
    print('Edge Points detection Precision: %f' % (pre_edge_precision / num_data))

    print('Corner Points detection Recall: %f' % (pre_corner_recall / num_data))
    print('Corner Points detection Precision: %f' % (pre_corner_precision / num_data))

    print('WireFrame Reconstruction Recall: %f' % (F_Recall/num_data))
    print('WireFrame Reconstruction Precision: %f' % (F_Precision/num_data))
    print('WireFrame Reconstruction F-Score: %f' % (WF_fsocre/num_data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line_num_point', type=int, default=30, help='Line Point Number [default: 30]')
    parser.add_argument('--angle_threshold', type=int, default=30,
                        help='The Angle threshold between each line [default: 15]')
    parser.add_argument('--edge_points_threshold', type=float, default=0.7,
                        help='Threshold of edge point prediction [default: 0.7]')
    parser.add_argument('--corner_points_threshold', type=float, default=0.95,
                        help='Threshold of corner point prediction [default: 0.95]')
    parser.add_argument('--line_threshold', type=float, default=0.01,
                        help='Threshold for line endpoints [default: 0.01]')
    parser.add_argument('--pso_ite_step', type=int, default=5,
                        help='The number of iterations of the PSO [default: 5]')
    parser.add_argument('--num_p', type=int, default=1000,
                        help='Number of particle swarm [default: 1000]')
    parser.add_argument('--scale', type=int, default=100,
                        help='A reduction of the speed range [default: 100]')
    parser.add_argument('--test_result_dir', default='test_result', help='The test result dir [default: test_result]')
    args = parser.parse_args()
    main(args)