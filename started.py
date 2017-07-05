import tensorflow as tf
sess = tf.Session()


W = tf.Variable([3], dtype=tf.float32)
b = tf.Variable([1.0],dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x:[1,2,3]}) )

