from __future__ import print_function
import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt

dirname = os.path.dirname
dataPath = dirname(dirname(os.getcwd()))+'\\DataSets\\'
pickle_file = dataPath + 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size*image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
def best_accuracy_id(accuracies):
    return accuracies.index(max(accuracies))

# Problem 1 Introduce L2 regularization
# 1.1 To logistic regression
#batch_size = 128
#graph = tf.Graph()
#with graph.as_default():
#    # Input data. For the training data, we use a placeholder that will be fed 
#    # at run time with a training minibatch.
#    tf_train_dataset = tf.placeholder(tf.float32,
#                                      shape = (batch_size, image_size*image_size))
#    tf_train_labels = tf.placeholder(tf.float32,
#                                     shape = (batch_size, num_labels))
#    tf_valid_dataset = tf.constant(valid_dataset)
#    tf_test_dataset = tf.constant(test_dataset)
#    beta = tf.placeholder(tf.float32)
    
#    # Variables
#    weights  = tf.Variable(
#        tf.truncated_normal([image_size*image_size, num_labels]))
#    biases = tf.Variable(
#        tf.zeros([num_labels]))

#    # Training computation
#    logits = tf.matmul(tf_train_dataset, weights)+biases
#    loss = tf.reduce_mean(
#        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + beta*tf.nn.l2_loss(weights)

#    # Optimizer
#    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#    # Predictions for training, validation and test data
#    train_prediction = tf.nn.softmax(logits)
#    valid_prediction = tf.nn.softmax(
#        tf.matmul(tf_valid_dataset, weights) + biases)
#    test_prediction = tf.nn.softmax(
#        tf.matmul(tf_test_dataset, weights) + biases)
#num_steps = 3001
#accuracies = []
#beta_val = [i for i in np.arange(0,0.01,0.0005)]
## Find the best beta_val
#for reg in beta_val:
#    with tf.Session(graph = graph) as session:
#        tf.global_variables_initializer().run()
#        print("Initialized")
#        for step in range(num_steps):
#            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#            batch_data = train_dataset[offset:offset+batch_size,:]
#            batch_labels = train_labels[offset:offset+batch_size, :]
#            # Prepare a dictionary telling the session where to feed the minibatch.
#            # The key of the dictionary is the placeholder node of the graph to be fed,
#            # and the value is the numpy array to feed to it
#            feed_dict = {tf_train_dataset: batch_data, tf_train_labels:batch_labels,
#                         beta: reg}
#            _,l,predictions = session.run(
#                [optimizer, loss, train_prediction], feed_dict=feed_dict)
#        accuracies.append(accuracy(test_prediction.eval(), test_labels))
#print('The max accuracy occurs at %f' % beta_val[best_accuracy_id(accuracies)])
#plt.semilogx(beta_val, accuracies)
#plt.grid(True)
#plt.title('Test Accuracy versus regularization Curve')
#plt.show()
#        #    if(step % 500==0):
#        #        print("Minibatch loss at step %d: %f" % (step, l))
#        #        print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#        #        print("Validation accuracy: %.1f%%" % accuracy(
#        #            valid_prediction.eval(), valid_labels))
#        #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
## Use the best beta to train the logistic regression
#with tf.Session (graph=graph) as session:
#    tf.global_variables_initializer().run()
#    print("Initialized")
#    for step in range(num_steps):
#         offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#         batch_data = train_dataset[offset:offset+batch_size,:]
#         batch_labels = train_labels[offset:offset+batch_size, :]
#         # Prepare a dictionary telling the session where to feed the minibatch.
#         # The key of the dictionary is the placeholder node of the graph to be fed,
#         # and the value is the numpy array to feed to it
#         feed_dict = {tf_train_dataset: batch_data, tf_train_labels:batch_labels,
#                      beta: beta_val[best_accuracy_id(accuracies)]}
#         _,l,predictions = session.run(
#             [optimizer, loss, train_prediction], feed_dict=feed_dict)
#         if(step % 500==0):
#            print("Minibatch loss at step %d: %f" % (step, l))
#            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#            print("Validation accuracy: %.1f%%" % accuracy(
#                valid_prediction.eval(), valid_labels))
#    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

## 1.2 To 1-layer neuro network
#num_hidden_nodes = 1024
#graph = tf.Graph()
#with graph.as_default():
#     # Input data. For the training data, we use a placeholder that will be fed 
#    # at run time with a training minibatch.
#    tf_train_dataset = tf.placeholder(tf.float32,
#                                      shape = (batch_size, image_size*image_size))
#    tf_train_labels = tf.placeholder(tf.float32,
#                                     shape = (batch_size, num_labels))
#    tf_valid_dataset = tf.constant(valid_dataset)
#    tf_test_dataset = tf.constant(test_dataset)
#    beta = tf.placeholder(tf.float32)
    
#    # Variables
#    weights_hidden = tf.Variable(
#        tf.truncated_normal([image_size*image_size, num_hidden_nodes]))
#    biases_hidden = tf.Variable(
#        tf.zeros([num_hidden_nodes]))
#    weights  = tf.Variable(
#        tf.truncated_normal([num_hidden_nodes, num_labels]))
#    biases = tf.Variable(
#        tf.zeros([num_labels]))

#    # Train computation
#    hidden_layer_train = tf.nn.relu(
#        tf.matmul(tf_train_dataset, weights_hidden)+biases_hidden)
#    logits   = tf.matmul(hidden_layer_train,weights)+biases
#    loss = tf.reduce_mean(
#        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
#            beta*(tf.nn.l2_loss(weights)+tf.nn.l2_loss(weights_hidden))
    
#    # Optimizer
#    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#    # Predictions for training, validation and test data
#    train_prediction = tf.nn.softmax(logits)
#    valid_hidden = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_hidden)+biases_hidden)
#    valid_prediction = tf.nn.softmax(
#        tf.matmul(valid_hidden, weights) + biases)
#    test_hidden = tf.nn.relu(tf.matmul(tf_test_dataset, weights_hidden)+biases_hidden)
#    test_prediction = tf.nn.softmax(
#        tf.matmul(test_hidden, weights) + biases)

## Find the best beta_val
#accuracies = []
#for reg in beta_val:
#    with tf.Session(graph = graph) as session:
#        tf.global_variables_initializer().run()
#        print("Initialized")
#        for step in range(num_steps):
#            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#            batch_data = train_dataset[offset:offset+batch_size,:]
#            batch_labels = train_labels[offset:offset+batch_size,:]
#            feed_dict = {tf_train_dataset:batch_data, tf_train_labels:batch_labels,
#                         beta:reg}
#            _,l,predictions = session.run(
#                [optimizer, loss, train_prediction], feed_dict = feed_dict)
#        accuracies.append(accuracy(test_prediction.eval(), test_labels))
#print('The max accuracy occurs at %f' % beta_val[best_accuracy_id(accuracies)])
#plt.semilogx(beta_val, accuracies)
#plt.grid(True)
#plt.title('Test Accuracy versus regularization Curve')
#plt.show()

## Use the best beta to train the NN
#with tf.Session(graph = graph) as session:
#    tf.global_variables_initializer().run()
#    print("Initialize")
#    for step in range(num_steps):
#        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#        batch_data = train_dataset[offset:offset+batch_size, :]
#        batch_labels = train_labels[offset:offset+batch_size,:]
#        feed_dict = {tf_train_dataset:batch_data, tf_train_labels:batch_labels,
#                     beta:beta_val[best_accuracy_id(accuracies)]}
#        _,l,predictions = session.run(
#            [optimizer, loss, train_prediction], feed_dict = feed_dict)
#        if(step % 500 ==0):
#            print("Minibatch loss at step %d: %f" % (step, l))
#            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#            print("Validation accuracy: %.1f%%" % accuracy(
#                valid_prediction.eval(), valid_labels))
#    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# Problem 2 Explore the overfitting: fewer batches make the performance bad
batch_size = 128
num_hidden_nodes = 1024
graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed 
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape = (batch_size, image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape = (batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta = tf.placeholder(tf.float32)
    
    # Variables
    weights_hidden = tf.Variable(
        tf.truncated_normal([image_size*image_size, num_hidden_nodes]))
    biases_hidden = tf.Variable(
        tf.zeros([num_hidden_nodes]))
    weights  = tf.Variable(
        tf.truncated_normal([num_hidden_nodes, num_labels]))
    biases = tf.Variable(
        tf.zeros([num_labels]))

    # Training computation
    hidden_layer_train = tf.nn.relu(
        tf.matmul(tf_train_dataset, weights_hidden)+biases_hidden)
    logits = tf.matmul(hidden_layer_train, weights)+biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for training, validation and test data
    train_prediction = tf.nn.softmax(logits)
    valid_hidden = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_hidden)+biases_hidden)
    valid_prediction = tf.nn.softmax(
        tf.matmul(valid_hidden, weights) + biases)
    test_hidden = tf.nn.relu(tf.matmul(tf_test_dataset, weights_hidden)+biases_hidden)
    test_prediction = tf.nn.softmax(
        tf.matmul(test_hidden, weights) + biases)
num_steps = 100
num_batches = 3
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        offset = step % num_batches
        batch_data = train_dataset[offset:offset+batch_size, :]
        batch_labels = train_labels[offset:offset+batch_size, :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels:batch_labels,
                     beta:0}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict = feed_dict)
        if(step %2 ==0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# Problem 3 Add the drop out to the 
batch_size = 128
num_hidden_nodes = 1024
graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed 
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape = (batch_size, image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape = (batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta = tf.placeholder(tf.float32)
    
    # Variables
    weights_hidden = tf.Variable(
        tf.truncated_normal([image_size*image_size, num_hidden_nodes]))
    biases_hidden = tf.Variable(
        tf.zeros([num_hidden_nodes]))
    weights  = tf.Variable(
        tf.truncated_normal([num_hidden_nodes, num_labels]))
    biases = tf.Variable(
        tf.zeros([num_labels]))

    # Training computation
    hidden_layer_train = tf.nn.relu(
        tf.matmul(tf_train_dataset, weights_hidden)+biases_hidden)
    drop1 = tf.nn.dropout(hidden_layer_train, 0.5)
    logits = tf.matmul(drop1, weights)+biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for training, validation and test data
    train_prediction = tf.nn.softmax(logits)
    valid_hidden = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_hidden)+biases_hidden)
    valid_prediction = tf.nn.softmax(
        tf.matmul(valid_hidden, weights) + biases)
    test_hidden = tf.nn.relu(tf.matmul(tf_test_dataset, weights_hidden)+biases_hidden)
    test_prediction = tf.nn.softmax(
        tf.matmul(test_hidden, weights) + biases)
num_steps = 100
num_batches = 3
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        offset = step % num_batches
        batch_data = train_dataset[offset:offset+batch_size, :]
        batch_labels = train_labels[offset:offset+batch_size, :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels:batch_labels,
                     beta:0}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict = feed_dict)
        if(step %2 ==0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# Problem 4 Try to get the best performance