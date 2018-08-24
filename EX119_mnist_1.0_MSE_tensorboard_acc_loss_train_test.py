# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import mnistdata
import math
print("Tensorflow version " + tf.__version__)
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(0)

# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 784])

# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# loss function: MSE
loss = tf.reduce_mean(tf.squared_difference(Y, Y_)) * 1000


# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)


# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("loss", loss)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# op to write logs to Tensorboard
train_writer = tf.summary.FileWriter('./logs/train', graph=tf.get_default_graph())
test_writer = tf.summary.FileWriter('./logs/test', graph=tf.get_default_graph())

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# run
for i in range(2000 + 1) :

	batch_X, batch_Y = mnist.train.next_batch(100)
	a, c, train_summary = sess.run([accuracy, loss, merged_summary_op], feed_dict={X : batch_X, Y_ : batch_Y})
	print("training : ", i, ' accuracy = ', '{:7.4f}'.format(a), ' loss = ', c)

	# Write logs at every iteration
	train_writer.add_summary(train_summary, i)

	a, c, test_summary = sess.run([accuracy, loss, merged_summary_op], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
	print("testing  : ",i, ' accuracy = ', '{:7.4f}'.format(a), ' loss = ', c)

	sess.run(train_step, feed_dict={X : batch_X, Y_ : batch_Y} )

	# Write logs at every iteration
	test_writer.add_summary(test_summary, i)
