from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True, validation_size=1)

import tensorflow as tf

W = tf.Variable(tf.zeros([64516,10]))
b = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, 64516])
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])

sess = tf.InteractiveSession()
saver = tf.train.Saver()

try:
    saver.restore(sess, "/tmp/trained-model.ckpt")
    print("Model restored.")
    images, labels = mnist.test.next_batch(100)
    print sess.run([W, b, y, y_], {x:images, y_:labels})
except Exception as e:
    print "Model has not been trained"