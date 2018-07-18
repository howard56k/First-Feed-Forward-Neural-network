import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
from create_sentiments import create_feature_sets_and_labels

train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

'''
input > weight > hidden layer 1 (activatico function) > weights > hidden l 2
(activation function) > weights > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch

'''

#mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3	= 500

n_classes = 2
batch_size = 100

#height x width
x = tf.placeholder('float',[None,len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
	#(input data * weights) + biases
#   my version
#tf.truncated_normal([784, n_nodes_hl1], stddev=0.1)
#tf.constant(0.1, shape=[shape])
	hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([len(train_x[0]), n_nodes_hl1], stddev=0.1)),
						'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1)),
						'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1)),
						'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev=0.1)),
						'biases':tf.Variable(tf.constant(0.1, shape=[n_classes]))}

#   Sentdex Version
	#hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
						#'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
#
	#hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
						#'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
#
	#hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
						#'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
#
	#output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
						#'biases':tf.Variable(tf.random_normal([n_classes]))}

	#(input data * weights) + biases

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#									learning rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#cycles feed forward + backprop
	hm_epochs = 10


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			i = 0
			while i < len(train_x):
				start = i
				end = i+ batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				i += batch_size

				_, c = sess.run([optimizer, cost], feed_dict = {x:batch_x, y:batch_y})
				epoch_loss += c
			print('Epoch: ', epoch+1, ' completed out of ', hm_epochs, ' loss: ', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)