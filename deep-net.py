import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data


'''
What's happening?
-------------------------------
input data > weight > hidden layer 1 (activation func) > weights 
	> hl2 (activation func) > weights > output layer

compare output to intended output (cost of loss function -- cross entropy)
optimizer tries to minimize cost (Adam, SGD, Adagrad) (backprop)

feed forward + backprop = epoch (10-20x) -- each cycle reduces cost function?

'''

#One hot (one is on, others off, electronics lingo)

mnist = input_data.read_data_sets("", one_hot=True)

#10 classes (0-9)
''' OHE format
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]

'''

#Nodes for hidden layers - these can change (narrow in middle?)
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

#Feed 100 images each time and manipuate weights
batch_size = 100

#Each image is a 28x28 1 row vector so 784 length unit vector
x = tf.placeholder('float', [None, 28*28]) #if shape, may throw error
y = tf.placeholder('float', [None, 10])

def neural_network_model(data):
	""" Creates weights based on random gaussian dist  """
	""" (input_data * Weights) + bias  (bias helps avoid having a zero output) """
	
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784,n_nodes_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer   = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
					  'biases': tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output




def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )

	# Has a learning_rate param default = 0.001 - not modified below
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 20 #how many feed fwd plus bkprop fixin weights (cycles of both)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		#Training the network Here
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size) #Typically define your own function
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch:', epoch, 'completed out of ', hm_epochs, 'loss: ', epoch_loss)

		# Once the weights are optimized, see accuracy
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))




train_neural_network(x)



