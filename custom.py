#before training, accuracy 0.07
#After training, accuracy 1

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=2)

import tensorflow as tf
sess = tf.InteractiveSession()

def correcting_function(row):
  condition = tf.logical_and(
      tf.equal(row[0], tf.reduce_max(row)),
      tf.less(row[0], 840))

  def swap_first_two(x):
    swapped_first_two = tf.stack([x[1], x[0]])
    rest = x[2:]
    return tf.concat([swapped_first_two, rest], 0)

  maybe_swapped = tf.cond(condition, lambda: swap_first_two(row), lambda: row)

  return maybe_swapped


W = tf.Variable(tf.zeros([64516,10]))
b = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, 64516])
actual_output = tf.matmul(x, W) + b
y = tf.map_fn(correcting_function, actual_output)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
optimizer = tf.train.GradientDescentOptimizer(0.02)
train = optimizer.minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()  
                                
for _ in range(1000):
    images, labels = mnist.train.next_batch(100)
    sess.run(train,{x:images, y_:labels})   

images, labels = mnist.test.next_batch(100)
print "New accuracy", sess.run([accuracy, y, y_], {x: images, y_: labels})
