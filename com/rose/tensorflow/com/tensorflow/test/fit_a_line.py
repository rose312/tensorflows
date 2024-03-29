import tensorflow as tf
import numpy as np


x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

print(x_data,y_data)
w = tf.Variable(tf.random_uniform([1],-0.1,1.0))
b = tf.Variable(tf.zeros([1]))

y = w * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(20000):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(w),sess.run(b))