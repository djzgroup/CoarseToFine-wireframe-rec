import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from utils import ABC_dataset_preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='ECR_Net', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log/ECR-Net', help='Log dir [default: log]')
parser.add_argument('--train_data', default='train_data', help='Dataset directory [default: train_data]')
parser.add_argument('--test_data', default='test_data', help='Dataset directory [default: test_data]')
parser.add_argument('--num_input_point', type=int, default=8096, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=101, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_INPUT_POINT = FLAGS.num_input_point
TRIAN_DATA = FLAGS.train_data
TEST_DATA = FLAGS.test_data
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of models def
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

TRAIN_DATASET = ABC_dataset_preprocess.ABCDataset(TRIAN_DATA, num_input_points=NUM_INPUT_POINT)
TEST_DATASET = ABC_dataset_preprocess.ABCDataset(TEST_DATA, num_input_points=NUM_INPUT_POINT, train=False)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def get_batch(dataset, idx, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_INPUT_POINT, 3))
    batch_edge_label = np.zeros((bsize, NUM_INPUT_POINT))
    batch_corner_label = np.zeros((bsize, NUM_INPUT_POINT))
    shuffle_idx = np.arange(NUM_INPUT_POINT)
    np.random.shuffle(shuffle_idx)
    for i in range(bsize):
        pcd, edge_label, corner_label = dataset[idx[i + start_idx]]
        pcd_center = np.mean(pcd, 0)  ### wether use normalization
        pcd -= pcd_center  ###
        batch_data[i, :, :] = pcd[shuffle_idx]
        batch_edge_label[i, :] = edge_label[shuffle_idx]
        batch_corner_label[i, :] = corner_label[shuffle_idx]
    return batch_data, batch_edge_label, batch_corner_label


def eval_one_epoch(sess, ops, test_writer):
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))

    num_data = len(TEST_DATASET)
    num_batches = (num_data + BATCH_SIZE - 1) // BATCH_SIZE
    total_loss = 0.0
    total_edge_loss = 0.0
    total_edge_acc = 0.0
    total_task1_acc = 0.0
    total_corner_loss = 0.0
    total_corner_acc = 0.0
    total_task2_acc = 0.0
    start_time = time.time()

    log_string(str(datetime.now()))
    log_string('---- EPOCH %3d EVALUATION ----' % (EPOCH_CNT))

    batch_data = np.zeros((BATCH_SIZE, NUM_INPUT_POINT, 3))
    batch_edge_label = np.zeros((BATCH_SIZE, NUM_INPUT_POINT))
    batch_corner_label = np.zeros((BATCH_SIZE, NUM_INPUT_POINT))

    for batch_idx in range(num_batches):
        begin_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx + 1) * BATCH_SIZE)
        cur_batch_size = end_idx - begin_idx
        cur_batch_data, cur_batch_edge_label, cur_batch_corner_label = get_batch(TEST_DATASET, test_idxs, begin_idx,
                                                                                 end_idx)
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_edge_label = cur_batch_edge_label
            batch_corner_label = cur_batch_corner_label
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_edge_label[0:cur_batch_size] = cur_batch_edge_label
            batch_corner_label[0:cur_batch_size] = cur_batch_corner_label

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_edge_p']: batch_edge_label,
                     ops['labels_corner_p']: batch_corner_label,
                     ops['is_training_pl']: is_training}
        summary, step, _, edge_loss, edge_acc, task1_acc, \
        corner_loss, corner_acc, task2_acc, loss = sess.run([ops['merged'],
                                                             ops['step'],
                                                             ops['train_op'],
                                                             ops['edge_loss'],
                                                             ops['edge_acc'],
                                                             ops['task1_acc'],
                                                             ops['corner_loss'],
                                                             ops['corner_acc'],
                                                             ops['task2_acc'],
                                                             ops['loss']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        total_loss += loss
        total_edge_loss += edge_loss
        total_edge_acc += edge_acc
        total_task1_acc += task1_acc
        total_corner_loss += corner_loss
        total_corner_acc += corner_acc
        total_task2_acc += task2_acc
    total_loss = total_loss * 1.0 / num_batches
    total_edge_loss = total_edge_loss * 1.0 / num_batches
    total_edge_acc = total_edge_acc * 1.0 / num_batches
    total_task1_acc = total_task1_acc * 1.0 / num_batches
    total_corner_loss = total_corner_loss * 1.0 / num_batches
    total_corner_acc = total_corner_acc * 1.0 / num_batches
    total_task2_acc = total_task2_acc * 1.0 / num_batches
    process_duration = time.time() - start_time
    examples_per_sec = num_data / process_duration
    sec_per_batch = process_duration / num_batches
    EPOCH_CNT += 1
    log_string('%s: step: %f loss: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
               % (datetime.now(), step, total_loss, process_duration, examples_per_sec, sec_per_batch))
    log_string('\tTesting Edge Mean_loss: %f' % total_edge_loss)
    log_string('\tTesting Edge Accuracy: %f' % total_edge_acc)
    log_string('\tTesting Task1 Accuracy: %f' % total_task1_acc)
    log_string('\tTesting Corner Mean_loss: %f' % total_corner_loss)
    log_string('\tTesting Corner Accuracy: %f' % total_corner_acc)
    log_string('\tTesting Task2 Accuracy: %f' % total_task2_acc)
    return total_loss


def train_one_epoch(sess, ops, train_writer):
    is_training = True

    # shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_data = len(TRAIN_DATASET)
    num_batches = num_data // BATCH_SIZE
    show_loss = 0.0
    total_loss = 0.0
    total_edge_loss = 0.0
    total_edge_acc = 0.0
    total_task1_acc = 0.0
    total_corner_loss = 0.0
    total_corner_acc = 0.0
    total_task2_acc = 0.0
    start_time = time.time()

    log_string(str(datetime.now()))

    for batch_idx in range(num_batches):
        begin_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_edge_label, batch_corner_label = get_batch(TRAIN_DATASET, train_idxs, begin_idx, end_idx)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_edge_p']: batch_edge_label,
                     ops['labels_corner_p']: batch_corner_label,
                     ops['is_training_pl']: is_training}
        summary, step, _, edge_loss, edge_acc, task1_acc, corner_loss, corner_acc, task2_acc, loss = sess.run(
            [ops['merged'],
             ops['step'],
             ops['train_op'],
             ops['edge_loss'],
             ops['edge_acc'],
             ops['task1_acc'],
             ops['corner_loss'],
             ops['corner_acc'],
             ops['task2_acc'],
             ops['loss']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        total_loss += loss
        show_loss += loss
        total_edge_loss += edge_loss
        total_edge_acc += edge_acc
        total_task1_acc += task1_acc
        total_corner_loss += corner_loss
        total_corner_acc += corner_acc
        total_task2_acc += task2_acc
        if (batch_idx + 1) % 10 == 0:
            log_string('-- %3d / %3d --' % (batch_idx + 1, num_batches))
            log_string('mean loss: %f' % (show_loss / 10))
            show_loss = 0
    total_loss = total_loss * 1.0 / num_batches
    total_edge_loss = total_edge_loss * 1.0 / num_batches
    total_edge_acc = total_edge_acc * 1.0 / num_batches
    total_task1_acc = total_task1_acc * 1.0 / num_batches
    total_corner_loss = total_corner_loss * 1.0 / num_batches
    total_corner_acc = total_corner_acc * 1.0 / num_batches
    total_task2_acc = total_task2_acc * 1.0 / num_batches
    process_duration = time.time() - start_time
    examples_per_sec = num_data / process_duration
    sec_per_batch = process_duration / num_batches
    log_string('%s: step: %f loss: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
               % (datetime.now(), step, total_loss, process_duration, examples_per_sec, sec_per_batch))
    log_string('\tTraining Edge Mean_loss: %f' % total_edge_loss)
    log_string('\tTraining Edge Accuracy: %f' % total_edge_acc)
    log_string('\tTraining Task1 Accuracy: %f' % total_task1_acc)
    log_string('\tTraining Corner Mean_loss: %f' % total_corner_loss)
    log_string('\tTraining Corner Accuracy: %f' % total_corner_acc)
    log_string('\tTraining Task2 Accuracy: %f' % total_task2_acc)


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_edge_p, labels_corner_p = MODEL.placeholder_inputs(BATCH_SIZE, NUM_INPUT_POINT)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            batch = tf.Variable(0, name='batch')
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)
            print("--- Get models and loss")

            # Get models and loss
            dof_feat = MODEL.get_feature(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            pred_labels_edge_p, pred_labels_corner_p = MODEL.get_task_feat(dof_feat, bn_decay=bn_decay)
            edge_loss, edge_acc, task1_acc, corner_loss, corner_acc, task2_acc, loss = MODEL.get_loss(pred_labels_edge_p, labels_edge_p, pred_labels_corner_p, labels_corner_p)
            tf.summary.scalar('edge loss', edge_loss)
            tf.summary.scalar('corner loss', corner_loss)
            tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variable
            saver = tf.train.Saver(max_to_keep=10)
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init Variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_edge_p': labels_edge_p,
               'labels_corner_p': labels_corner_p,
               'is_training_pl': is_training_pl,
               'pred_labels_edge_p': pred_labels_edge_p,
               'pred_labels_corner_p': pred_labels_corner_p,
               'edge_loss': edge_loss,
               'edge_acc': edge_acc,
               'task1_acc': task1_acc,
               'corner_loss': corner_loss,
               'corner_acc': corner_acc,
               'task2_acc': task2_acc,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        least_total_loss = 100
        for epoch in range(MAX_EPOCH):
            log_string('**** TRAIN EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            train_one_epoch(sess, ops, train_writer)
            total_loss = eval_one_epoch(sess, ops, test_writer)
            if total_loss < least_total_loss:
                least_total_loss = total_loss
                model_final_path = "final.ckpt"
                save_final_path = saver.save(sess, os.path.join(LOG_DIR, model_final_path))
            # Save the variable to disk
            model_ccc_path = "model" + str(epoch) + ".ckpt"
            save_path = saver.save(sess, os.path.join(LOG_DIR, model_ccc_path))
            log_string("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    log_string('pid: %s' % (str(os.getgid())))
    train()
    LOG_FOUT.close()
