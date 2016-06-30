#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


learning_rate = 0.001
traing_approach = 30
batch_size = 100
default_node_size = 250
display_step = 1

X = tf.placeholder("float",[None,784])
Y = tf.placeholder("float",[None,10])

W1 = tf.Variable(tf.random_normal([784,default_node_size]))
W2 = tf.Variable(tf.random_normal([default_node_size,default_node_size]))
W3 = tf.Variable(tf.random_normal([default_node_size,10]))

B1= tf.Variable(tf.random_normal([default_node_size]))
B2 = tf.Variable(tf.random_normal([default_node_size]))
B3 = tf.Variable(tf.random_normal([10]))


L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),B2))
hyp = tf.add(tf.matmul(L2,W3),B3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hyp,Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    for eproch in range(traing_approach):
        avg_cost = 0
        total_branch = int(mnist.train.num_examples/batch_size)

        for i in range(total_branch):
            batch_xs, batch_ys =mnist.train.next_batch(batch_size)
            session.run(optimizer,feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost += session.run(cost, feed_dict={X:batch_xs,Y:batch_ys})/total_branch

        if(eproch % display_step ==0):
            print "Eproch","%04d" % (eproch+1), "cost:","{:.9f}".format(avg_cost)



    print("optimization finished")

    #Test Model
    correct_prediction = tf.equal(tf.argmax(hyp,1),tf.argmax(Y,1))
    #accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print "accuracy", accuracy.eval({X:mnist.test.images,Y:mnist.test.labels})
