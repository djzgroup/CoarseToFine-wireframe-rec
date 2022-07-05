import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

def placeholder_inputs(batch_size, numpoint):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, numpoint, 3))
    labels_edge_p = tf.compat.v1.placeholder(tf.int32, shape=(batch_size, numpoint))
    labels_corner_p = tf.compat.v1.placeholder(tf.int32, shape=(batch_size, numpoint))
    return pointclouds_pl, labels_edge_p, labels_corner_p

def get_feature(point_cloud, is_training, bn_decay=None):
    l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
    l0_points = tf.slice(point_cloud, [0, 0, 3], [-1, -1, 0])

    # set abstraction layers
    with tf.compat.v1.variable_scope('set_abstraction', reuse=tf.compat.v1.AUTO_REUSE):
        # layer 1
        l1_xyz, l1_points, _ = pointnet_sa_module(l0_xyz, l0_points, npoint=4096, radius=0.05, nsample=32, mlp=[32, 32, 64],
                                                  mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay,
                                                  scope='layer1')
        # layer 2
        l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points, npoint=2048, radius=0.1, nsample=32, mlp=[64, 64, 128],
                                                  mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay,
                                                  scope='layer2')
        # layer 3
        l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=1024, radius=0.2, nsample=32,
                                                  mlp=[128, 128, 256], mlp2=None, group_all=False, is_training=is_training,
                                                  bn_decay=bn_decay, scope='layer3')
        # layer 4
        l4_xyz, l4_points, _ = pointnet_sa_module(l3_xyz, l3_points, npoint=512, radius=0.4, nsample=32,
                                                  mlp=[256, 256, 512], mlp2=None, group_all=False, is_training=is_training,
                                                  bn_decay=bn_decay, scope='layer4')

    # feature propagation layers
    with tf.compat.v1.variable_scope('feature_propagation', reuse=tf.compat.v1.AUTO_REUSE):
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], is_training, bn_decay,
                                       scope='fp_layer1')
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], is_training, bn_decay,
                                       scope='fp_layer2')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 256], is_training, bn_decay,
                                       scope='fp_layer3')
        l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], is_training, bn_decay,
                                       scope='fp_layer4')

    # FC layers
    with tf.compat.v1.variable_scope('fc_layer', reuse=tf.compat.v1.AUTO_REUSE):
        net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='pointnet/fc1',
                             bn_decay=bn_decay)
        # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='pointnet/dp1')
        dof_feat = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='pointnet/fc_dof',
                                  bn_decay=bn_decay)

    return dof_feat

def get_task_feat(dof_feat, bn_decay=None):
    # task1: edge_point
    with tf.compat.v1.variable_scope('edge_net', reuse=tf.compat.v1.AUTO_REUSE):
        feat1 = tf_util.conv1d(dof_feat, 64, 1, padding='VALID', activation_fn=None, scope='task1/fc1', bn_decay=bn_decay)
        pred_labels_edge_p = tf_util.conv1d(feat1, 2, 1, padding='VALID', activation_fn=None, scope='task1/fc2',
                                         bn_decay=bn_decay)

    # task: corner_point
    with tf.compat.v1.variable_scope('corner_net', reuse=tf.compat.v1.AUTO_REUSE):
        feat2 = tf_util.conv1d(dof_feat, 64, 1, padding='VALID', activation_fn=None, scope='task2/fc1', bn_decay=bn_decay)
        pred_labels_corner_p = tf_util.conv1d(feat2, 2, 1, padding='VALID', activation_fn=None, scope='task2/fc2',
                                          bn_decay=bn_decay)
    return pred_labels_edge_p, pred_labels_corner_p


def get_loss(pred_labels_edge_p, labels_edge_p, pred_labels_corner_p, labels_corner_p):
    num_point = pred_labels_edge_p.get_shape()[1].value

    # task1_loss
    mask_edge = tf.cast(labels_edge_p, tf.float32)
    neg_mask_edge = tf.ones_like(mask_edge) - mask_edge
    Np_edge = tf.expand_dims(tf.reduce_sum(mask_edge, axis=1), 1)
    Ng_edge = tf.expand_dims(tf.reduce_sum(neg_mask_edge, axis=1), 1)

    edge_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred_labels_edge_p, labels=labels_edge_p) * (mask_edge * (Ng_edge / Np_edge) + 1))
    # edge_loss = tf.reduce_mean(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_labels_edge_p, labels=labels_edge_p))
    edge_acc = tf.reduce_mean(tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(pred_labels_edge_p, axis=2, output_type=tf.int32), labels_edge_p),
                tf.float32) * mask_edge, axis=1) / tf.reduce_sum(mask_edge, axis=1))
    # edge_precision = tf.reduce_mean(tf.reduce_sum(
    #     tf.cast(tf.equal(tf.argmax(pred_labels_edge_p, axis=2, output_type=tf.int32), labels_edge_p),
    #             tf.float32) * mask_edge, axis=1) / (tf.reduce_sum(tf.cast(tf.argmax(pred_labels_edge_p, axis=2, output_type=tf.int32),tf.float32),axis=1) + 1e-12))
    task1_acc = tf.reduce_mean(tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(pred_labels_edge_p, axis=2, output_type=tf.int32), labels_edge_p),
                tf.float32),axis=1) / num_point)

    # task2_loss
    mask_corner = tf.cast(labels_corner_p, tf.float32)
    neg_mask_corner = tf.ones_like(mask_corner) - mask_corner
    Np_corner = tf.expand_dims(tf.reduce_sum(mask_corner, axis=1), 1)
    Ng_corner = tf.expand_dims(tf.reduce_sum(neg_mask_corner, axis=1), 1)

    corner_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred_labels_corner_p, labels=labels_corner_p) * (mask_corner * (Ng_corner / Np_corner) + 1))
    # corner_loss = tf.reduce_mean(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_labels_corner_p, labels=labels_corner_p))
    corner_acc = tf.reduce_mean(tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(pred_labels_corner_p, axis=2, output_type=tf.int32), labels_corner_p),
                tf.float32) * mask_corner, axis=1) / tf.reduce_sum(mask_corner, axis=1))
    # corner_precision = tf.reduce_mean(tf.reduce_sum(
    #     tf.cast(tf.equal(tf.argmax(pred_labels_corner_p, axis=2, output_type=tf.int32), labels_corner_p),
    #             tf.float32) * mask_corner, axis=1) / (tf.reduce_sum(tf.cast(tf.argmax(pred_labels_corner_p, axis=2, output_type=tf.int32),tf.float32),axis=1) + 1e-12))
    task2_acc = tf.reduce_mean(tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(pred_labels_corner_p, axis=2, output_type=tf.int32), labels_corner_p),
                tf.float32), axis=1) / num_point)

    w1 = 1
    w2 = 1

    loss = edge_loss * w1 + corner_loss * w2

    tf.summary.scalar('all loss', loss)
    tf.summary.scalar('losses', loss)
    # return edge_loss, edge_recall, edge_precision, edge_acc, corner_loss, corner_recall, corner_precision, corner_acc, loss
    return edge_loss, edge_acc, task1_acc, corner_loss, corner_acc, task2_acc, loss