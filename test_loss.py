import tensorflow as tf
import numpy as np

graph = tf.Graph()
with graph.as_default():
    # Placeholders to take in batches onf data
    tf_label = tf.placeholder(dtype=tf.int32, shape=[None])
    tf_logits = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    tf_mask = tf.placeholder(dtype=tf.float32, shape=[None])

    one_hot_label = tf.one_hot(tf_label, 10, on_value=1.0, off_value=0.0)

    loss_1 = tf.losses.softmax_cross_entropy(one_hot_label, tf_logits, weights=tf_mask)
    loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_label, logits=tf_logits)
    loss_3 = tf.losses.sparse_softmax_cross_entropy(labels=tf_label, logits=tf_logits)


with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())

    label = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    mask = np.ones((10))
    logits = np.random.rand(10, 10)

    feed_dict={tf_label: label, tf_logits: logits, tf_mask: mask}
    loss_1, loss_2, loss_3 = session.run([loss_1, loss_2, loss_3], feed_dict=feed_dict)

    print("tf.losses.softmax_cross_entropy:")
    print(loss_1)
    print("tf.nn.sparse_softmax_cross_entropy_with_logits:")
    print(loss_2)
    print("tf.losses.sparse_softmax_cross_entropy:")
    print(loss_3)
