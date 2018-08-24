import tensorflow as tf
import mnistdata
import matplotlib.pyplot as plt
import math
import numpy as np
# 학습데이터 가져오기 
mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

# 매개변수 W와 b를 초기화 
K = 6
L = 12
M = 24 
N = 200 

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones((M))/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7* M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)


# 학습 데이터를 주입할 place holder를 정의  
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
pkeep = tf.placeholder(tf.float32)
step = tf.placeholder(tf.int32)
# cnn 모델 정의 
stride = 1 # padding을 SAME으로 줘서 필터 크기를 맞춰주었당
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
Y1d = tf.nn.dropout(Y1, pkeep)
stride = 2
Y2 = tf.nn.relu(tf.nn.conv2d(Y1d, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
Y2d = tf.nn.dropout(Y2, pkeep)
stride = 2
Y3 = tf.nn.relu(tf.nn.conv2d(Y2d, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

YY = tf.reshape(Y3d, shape=[-1, 7 * 7 * M])
Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)

# 손실함수 정의 
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 손실갑 최소화하는 optimization 방법 정의 
learning_rate = 0.001 + tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

tf.summary.scalar("loss", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())



#sess 객체 생성 및 변수 초기화 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_acc_list = []
test_acc_list = []
train_loss_list = []
test_loss_list = []

for i in range(2000+1):
	# 샘플링 
	batch_X, batch_Y = mnist.train.next_batch(100)
	# train 정확도
	a, c, summary= sess.run([accuracy, cross_entropy, merged_summary_op], feed_dict={X: batch_X, Y_:batch_Y, pkeep: 1.0, step: i})
	print("training: ", i, a, c)

	summary_writer.add_summary(summary, i)

	if i % 10 == 0:
		train_acc_list.append(a)
		train_loss_list.append(c)

	if i % 10 == 0:
		#test 정확도 
		a, c, summary = sess.run([accuracy, cross_entropy, merged_summary_op], feed_dict={X: mnist.test.images, Y_:mnist.test.labels, pkeep: 1.0})
		test_acc_list.append(a)
		test_loss_list.append(c)
		summary_writer.add_summary(summary, i)

	# 학습 
	sess.run(train_step, {X: batch_X, Y_: batch_Y, pkeep: 0.75, step: i})

# draw graph : accuracy
x = np.arange(len(train_acc_list))
plt.figure(1) 
plt.plot(x, train_acc_list,  label='train', markevery=1)
plt.plot(x, test_acc_list, label='test', markevery=1)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
# plt.show()

# draw graph : loss
x = np.arange(len(train_loss_list))
plt.figure(2) 
plt.plot(x, train_loss_list,  label='train', markevery=1)
plt.plot(x, test_loss_list, label='test', markevery=1)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 100)
plt.legend(loc='upper right')
plt.show()