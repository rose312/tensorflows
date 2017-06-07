import  tensorflow as tf
#定义 符号 变量 也称为占位符
a = tf.placeholder("float32")
b = tf.placeholder("float32")
#构造一个op节点
y = tf.mul(a,b)
z = tf.div(a,b)
h = tf.shape(a)
#建立会话
sess = tf.Session()
#运行会话，输入数据，并计算节点 同时打印结果
print(sess.run(y ,feed_dict={a:3.0,b:3.0}))
print(sess.run(h ,feed_dict={a:3.0,b:3.0}))

#任务完成 关闭会话
sess.close();