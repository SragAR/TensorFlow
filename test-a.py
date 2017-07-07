

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=0)

import tensorflow as tf
sess = tf.InteractiveSession()

y = tf.placeholder(tf.float32, [None, 10])

images, labels = mnist.train.next_batch(100)
print sess.run(y, {y: labels})