import tensorflow as tf

node1 = tf.constant(3.0)
node2 = tf.constant(4.0, dtype=tf.float32)
node3 = tf.add(node1, node2)

sess = tf.Session()
print node1, node2, node3
print(sess.run([node1,node2]))
print(sess.run(node3))

#parameterized

a =tf.placeholder(tf.float32)
b= tf.placeholder(tf.float32)
adder_node = a + b #shorthand for tf.add(a,b)
add_and_triple = adder_node * 3

print(sess.run(adder_node, {a:5, b:3.5}))
print(adder_node, {a:[1,2], b:[5,6]})
print(sess.run(add_and_triple, {a:2, b:6}))

