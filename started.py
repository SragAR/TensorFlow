#Linear regression program 

import tensorflow as tf
sess = tf.Session()
#name param important for saving variables`
W = tf.Variable([3], dtype=tf.float32, name="W")
b = tf.Variable([-3],dtype=tf.float32, name="b")
saver = tf.train.Saver()

x = tf.placeholder(tf.float32)
# y is the desired output
y = tf.placeholder(tf.float32)
linear_model = W * x + b
squared_deltas = tf.square(linear_model - y)
#loss functin
loss = tf.reduce_sum(squared_deltas)
#initialize variables
init = tf.global_variables_initializer()
sess.run(init)

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
old_W, old_b, old_loss = sess.run([W, b, loss], {x:[1,2,3], y:[0, -1, -2]})
print("old W: %s \told b: %s\told loss : %s"%(old_W, old_b, old_loss))

for i in range(1000):
    sess.run(train,{x:[1,2,3], y:[0, -1, -2]})
    if i %100 == 0:
        save_path = saver.save(sess, "/tmp/model"+str(i)+".ckpt")
        print("Model saved in file: %s" % save_path)

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:[1,2,3], y:[0, -1, -2]})
print("new W: %s \tnew b: %s\tnew loss : %s"%(curr_W, curr_b, curr_loss))