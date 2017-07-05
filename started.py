import tensorflow as tf
sess = tf.Session()


W = tf.Variable([3], dtype=tf.float32)
b = tf.Variable([-3],dtype=tf.float32)
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

#assigning value to variable 
fixW = tf.assign(W,[-1])
fixb = tf.assign(b, [1])
sess.run([fixW, fixb])

print(sess.run(linear_model, {x:[1,2,3], y:[4, 7, 10]}))
print sess.run(loss,{x:[1,2,3], y:[0, -1, -2]})
