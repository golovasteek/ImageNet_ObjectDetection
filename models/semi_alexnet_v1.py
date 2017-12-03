import tensorflow as tf


def _semi_alexnet_v1(data, output_labels, is_training):
    conv = data
    conv = tf.layers.conv2d(
        inputs=conv,
        filters=48,
        padding='valid',
        kernel_size=11,
        strides=4,
        activation=tf.nn.relu
    )
    print conv.shape.as_list()
    conv = tf.layers.conv2d(
        inputs=conv,
        filters=128,
        padding='valid',
        kernel_size=5,
        strides=1,
        activation=tf.nn.relu
    )
    conv = tf.layers.max_pooling2d(
        conv,
        pool_size=[5, 5],
        strides=2
    )
    print conv.shape.as_list()

    conv = tf.layers.conv2d(
        inputs=conv,
        filters=192,
        padding='same',
        kernel_size=3,
        strides=1,
        activation=tf.nn.relu
    )
    conv = tf.layers.max_pooling2d(
        conv,
        pool_size=[3, 3],
        strides=2
    )
    print conv.shape.as_list()

    conv = tf.layers.conv2d(
        inputs=conv,
        filters=192,
        padding='same',
        kernel_size=3,
        strides=1,
        activation=tf.nn.relu
    )
    print conv.shape.as_list()

    conv = tf.layers.conv2d(
        inputs=conv,
        filters=128,
        padding='same',
        kernel_size=3,
        strides=1,
        activation=tf.nn.relu
    )

    shape = conv.shape.as_list()
    print shape
    print [-1, shape[1] * shape[2] * shape[3]]
    pool_flat = tf.reshape(conv, [-1, shape[1] * shape[2] * shape[3]])
    dense = tf.layers.dense(
        inputs=pool_flat,
        units=2048,
        activation=tf.nn.relu
    )

    print dense.shape.as_list()
    dense = tf.layers.dense(
        inputs=dense,
        units=2048,
        activation=tf.nn.relu
    )
    print dense.shape.as_list()
    logits = tf.layers.dense(
        inputs=dense,
        units=output_labels
    )
    return logits

def semi_alexnet_v1(*args):
    with tf.name_scope("semi_alex_v1"):
        return _semi_alexnet_v1(*args)