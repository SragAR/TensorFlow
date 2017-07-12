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
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

tf.global_variables_initializer().run()

for _ in range(1000):
    images, labels = mnist.train.next_batch(1)
    sess.run(train,{x:images, y_:labels})   

save_path = saver.save(sess, "/tmp/trained-model.ckpt")
