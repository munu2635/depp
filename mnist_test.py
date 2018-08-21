import tensorflow as tf
import mnistdata 
import matplotlib.pyplot as plt
import numpy as np

mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

XX = tf.reshape(X, [-1, 784])
Y = tf.nn.softmax(tf.matmul(XX,W) + b)

#datavis = tensorflowvisu.MnistDataVis()

loss = tf.reduce_mean(tf.squared_difference(Y, Y_)) * 1000

train_step = tf.train.GradientDescentOptimizer(0.005). minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accyracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_acc_list= []
train_loss_list = []
test_acc_list = []
test_loss_list = []

for i in range(2000+1):

	batch_X, batch_Y = mnist.train.next_batch(100)
	a, c = sess.run([accyracy, loss], feed_dict = {X: batch_X, Y_: batch_Y })

	sess.run(train_step, feed_dict={X:batch_X, Y_:batch_Y})
	print("training : ", i, " accyracy : ", a, " loss : ", c)

	at, ct = sess.run([accyracy, loss], feed_dict = {X: mnist.test.images, Y_: mnist.test.labels })
	sess.run(train_step, feed_dict={X:batch_X, Y_:batch_Y})

	train_acc_list.append(a)
	train_loss_list.append(c)
	test_acc_list.append(at)
	test_loss_list.append(ct)

i = np.arange(0, 2001, 1)
plt.figure(1)
plt.title('accyracy')
plt.plot(i, train_acc_list , "b-")
plt.plot(i, test_acc_list, "r")

plt.figure(2)
plt.title('loss')
plt.plot(i, train_loss_list , "b-")
plt.plot(i, test_loss_list, "r")
plt.show()