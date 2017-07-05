#before training, accuracy 0.07
#After training, accuracy 0.13

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, 784])
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])
error = tf.square(y - y_)
loss = tf.reduce_sum(error)

tf.global_variables_initializer().run()
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
images, labels = mnist.train.next_batch(100)

correct_prediction =  tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print"Old accuracy"
#print(sess.run([W, b]))
print sess.run(accuracy, {x: images, y_: labels})
tf.global_variables_initializer().run()                                  
for _ in range(21):
    sess.run(train,{x:images, y_:labels})   

print"New accuracy"
# print(sess.run([W, b]))
# print sess.run(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)),{x:images, y_:labels})
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
print sess.run(accuracy, {x: images, y_: labels})
