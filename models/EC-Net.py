import tensorflow as tf
from utils import tf_util
from utils.pointnet_util import pointnet_sa_module, pointnet_fp_module

def placeholder_inputs(batch_size, numpoint):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, numpoint, 3))
    labels_edge_p = tf.placeholder(tf.int32, shape=(batch_size, numpoint))
    labels_corner_p = tf.placeholder(tf.int32, shape=(batch_size, numpoint))
    return pointclouds_pl, labels_edge_p, labels_corner_p

def get_gen_model(point_cloud, is_training, bradius = 1.0, use_bn = False, use_normal = False, bn_decay = None, up_ratio = 4, idx=None, is_crop=False):

    print("Crop flag is ",is_crop)

    with tf.variable_scope('EC-Net',reuse=tf.AUTO_REUSE) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        l0_xyz = point_cloud[:,:,0:3]
        if use_normal:
            l0_points = point_cloud[:,:,3:]
        else:
            l0_points = None
        # Layer 1
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=num_point, radius=bradius*0.1,bn=use_bn,
                                                           nsample=12, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer1')

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point/2, radius=bradius*0.2,bn=use_bn,
                                                           nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer2')

        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point/4, radius=bradius*0.4,bn=use_bn,
                                                           nsample=32, mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer3')

        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point/8, radius=bradius*0.6,bn=use_bn,
                                                           nsample=32, mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # Feature Propagation layers

        up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
                                       scope='fa_layer1',bn=use_bn)

        up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
                                       scope='fa_layer2',bn=use_bn)

        up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay,
                                       scope='fa_layer3',bn=use_bn)

        feat = tf.concat([up_l4_points, up_l3_points, up_l2_points, l1_points], axis=-1)
        feat = tf.expand_dims(feat, axis=2)

        #the new generate points
        up_feat = tf_util.conv2d(feat, 256, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv1_1d', bn_decay=bn_decay)

        up_feat = tf_util.conv2d(up_feat, 128, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=use_bn, is_training=is_training,
                                 scope='conv2_1d',
                                 bn_decay=bn_decay)

        # predict the edge points
        feat_edge_p = tf_util.conv2d(up_feat, 64, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=False, is_training=is_training,
                                    scope='edge_fc1', bn_decay=bn_decay, weight_decay=0.0)
        pred_labels_edge_p = tf_util.conv2d(feat_edge_p, 2, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='edge_fc2', bn_decay=bn_decay,
                                 activation_fn=None, weight_decay=0.0)
        # predict the corner points
        feat_corner_p = tf_util.conv2d(up_feat, 64, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='corner_fc1', bn_decay=bn_decay, weight_decay=0.0)
        pred_labels_corner_p = tf_util.conv2d(feat_corner_p, 2, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=False, is_training=is_training,
                                            scope='corner_fc2', bn_decay=bn_decay,
                                            activation_fn=None, weight_decay=0.0)
        pred_labels_edge_p = tf.squeeze(pred_labels_edge_p, [2])

        pred_labels_corner_p = tf.squeeze(pred_labels_corner_p, [2])

    return pred_labels_edge_p, pred_labels_corner_p

def get_loss(pred_labels_edge_p, labels_edge_p, pred_labels_corner_p, labels_corner_p):
    num_point = pred_labels_edge_p.get_shape()[1].value

    # edge_loss
    mask_edge = tf.cast(labels_edge_p, tf.float32)

    edge_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred_labels_edge_p, labels=labels_edge_p))
    edge_acc = tf.reduce_mean(tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(pred_labels_edge_p, axis=2, output_type=tf.int32), labels_edge_p),
                tf.float32) * mask_edge, axis=1) / tf.reduce_sum(mask_edge, axis=1))
    task1_acc = tf.reduce_mean(tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(pred_labels_edge_p, axis=2, output_type=tf.int32), labels_edge_p),
                tf.float32), axis=1) / num_point)

    # corner_loss
    mask_corner = tf.cast(labels_corner_p, tf.float32)

    corner_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred_labels_corner_p, labels=labels_corner_p))
    corner_acc = tf.reduce_mean(tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(pred_labels_corner_p, axis=2, output_type=tf.int32), labels_corner_p),
                tf.float32) * mask_corner, axis=1) / tf.reduce_sum(mask_corner, axis=1))
    task2_acc = tf.reduce_mean(tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(pred_labels_corner_p, axis=2, output_type=tf.int32), labels_corner_p),
                tf.float32), axis=1) / num_point)

    w1 = 1
    w2 = 1

    loss = edge_loss * w1 + corner_loss * w2

    tf.summary.scalar('all loss', loss)
    tf.summary.scalar('losses', loss)
    return edge_loss, edge_acc, task1_acc, corner_loss, corner_acc, task2_acc, loss