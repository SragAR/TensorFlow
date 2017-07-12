
import tensorflow as tf
sess = tf.Session()
#name for variables very important
W = tf.Variable([3], dtype=tf.float32,name="W")
b = tf.Variable([-3],dtype=tf.float32,name="b")
saver = tf.train.Saver()
#Initialization not required before loading saved variables
#Here it is done to give defaukt value in case of error
init = tf.global_variables_initializer()
sess.run(init)
#loading stored variables 
try:
    saver.restore(sess, "/tmp/model500.ckpt")
    print("Model restored.")
except Exception as e:
    pass


print(sess.run([W, b ]))