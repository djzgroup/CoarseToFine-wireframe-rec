import argparse
import numpy as np
import tensorflow as tf
from glob import glob
import importlib
import os
import sys
from tqdm import tqdm
curr_dir = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, ''))

def predict(pointcloud, args):
    inputs = tf.placeholder(tf.float32, (1, args.num_point, 3))
    is_training = False
    model = importlib.import_module(args.models)

    dof_feat = model.get_feature(inputs, is_training, bn_decay=None)
    pred_labels_edge_p, pred_labels_corner_p = model.get_task_feat(dof_feat, bn_decay=None)
    # pred_labels_edge_p, pred_labels_corner_p = model.get_gen_model(inputs, is_training, bradius=1.0, bn_decay=None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    pred_edge_labels, pred_corner_labels = sess.run([pred_labels_edge_p, pred_labels_corner_p], feed_dict={inputs: [pointcloud]})
    pred_edge_labels = np.squeeze(pred_edge_labels)
    pred_corner_labels = np.squeeze(pred_corner_labels)
    return pred_edge_labels, pred_corner_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='ECR_Net', help='Model name [default: models]')
    parser.add_argument('--num_point', type=int, default=8096, help='Point Number [default: 2048]')
    parser.add_argument('--checkpoint', default='log/ECR-Net/model100.ckpt', help='model checkpoint file path [default: log/model100.ckpt]')
    args = parser.parse_args()
    save_to_folder1 = os.path.join(curr_dir, 'test_result/ECR-Net/test_datasets')
    save_to_folder2 = os.path.join(curr_dir, 'test_result/ECR-Net/test_PC2WF_datasets')
    if not os.path.exists(save_to_folder1):
        os.makedirs(save_to_folder1)
    if not os.path.exists(save_to_folder2):
        os.makedirs(save_to_folder2)
    test_file_list = glob(os.path.join(curr_dir, f'../PCP2WF/PC2WF_data/*.npz'))
    test_file_list.sort()
    for test_file in tqdm(test_file_list):
        file_name = (test_file.split('/')[-1]).split('.')[0]
        data = np.load(test_file)
        data_size = len(data)
        pointcloud = data['pointcloud']
        pred_labels_edge_p, pred_labels_corner_p = predict(pointcloud, args)
        if data_size == 3:
            np.savez_compressed(save_to_folder1 + '/' + file_name,
                                pointcloud=pointcloud,
                                edge_label=data['labels_edge_p'],
                                corner_label=data['labels_corner_p'],
                                pred_labels_edge_p=pred_labels_edge_p,
                                pred_labels_corner_p=pred_labels_corner_p)
        elif data_size == 1:
            np.savez_compressed(save_to_folder2 + '/' + file_name,
                                pointcloud=pointcloud,
                                pred_labels_edge_p=pred_labels_edge_p,
                                pred_labels_corner_p=pred_labels_corner_p)