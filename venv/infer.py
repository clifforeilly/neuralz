from tensorflow.python.platform import gfile
import tensorflow as tf
from keras.preprocessing import image
import numpy as np


graph2 = tf.Graph()
with graph2.as_default():
    with tf.Session(graph=graph2) as sess:
        # Restore saved values
        print('\nRestoring...')
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            "E:/1/"
        )
        print('Ok')

        np.random.seed(0)
        features = np.random.randn(64, 10, 30)
        labels = np.eye(5)[np.random.randint(0, 5, (64,))]

        for op in graph2.get_operations():
            print(op.name)

        labels_data_ph = graph2.get_tensor_by_name('labels_data_ph:0')
        features_data_ph = graph2.get_tensor_by_name('features_data_ph:0')
        batch_size_ph = graph2.get_tensor_by_name('batch_size_ph:0')
        # Get restored model output
        restored_logits = graph2.get_tensor_by_name('dense/BiasAdd:0')
        # Get dataset initializing operation
        dataset_init_op = graph2.get_operation_by_name('init')

        # Initialize restored dataset
        sess.run(
            dataset_init_op,
            feed_dict={
                features_data_ph: features,
                labels_data_ph: labels,
                batch_size_ph: 32
            }
        )
        # Compute inference for both batches in dataset
        restored_values = []
        for i in range(2):
            restored_values.append(sess.run(restored_logits))
            print('Restored values: ', restored_values[i][0])
