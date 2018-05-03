import tensorflow as tf
import numpy as np


labels = np.array([[1,1,1,0],
                   [1,1,1,0],
                   [1,1,1,0],
                   [1,1,1,0]], dtype=np.uint8)

predictions = np.array([[1,1,0,1],
                        [1,1,0,1],
                        [1,1,1,0],
                        [1,1,1,0]], dtype=np.uint8)

n_batches = len(labels)

graph = tf.Graph()
with graph.as_default():
    # Placeholders to take in batches onf data
    tf_label = tf.placeholder(dtype=tf.int32, shape=[None])
    tf_prediction = tf.placeholder(dtype=tf.int32, shape=[None])

    # Define the metric and update operations
    tf_metric, tf_metric_update = tf.metrics.accuracy(tf_label,
                                                      tf_prediction,
                                                      name="my_metric")

    # Isolate the variables stored behind the scenes by the metric operation
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")

    # Define initializer to initialize/reset running variables
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    init_all = tf.local_variables_initializer()


with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())

    # initialize/reset the running variables
    session.run(running_vars_initializer)

    for i in range(2):
        # Update the running variables on new batch of samples
        feed_dict={tf_label: labels[i], tf_prediction: predictions[i]}
        session.run(tf_metric_update, feed_dict=feed_dict)

    # Calculate the score
    score = session.run(tf_metric)
    print("[TF] SCORE: ", score)

    # initialize/reset the running variables
    tf.get_default_session().run(running_vars_initializer)
    score = session.run(tf_metric)
    print("[TF] SCORE: ", score)

    for i in range(2):
        # Update the running variables on new batch of samples
        feed_dict={tf_label: labels[i + 1], tf_prediction: predictions[i + 2]}
        session.run(tf_metric_update, feed_dict=feed_dict)

    # Calculate the score
    score = session.run(tf_metric)
    print("[TF] SCORE: ", score)
