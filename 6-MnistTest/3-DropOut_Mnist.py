#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def xavier_init(input_size, output_size, uniform=True):
    if uniform:
        init_range= tf.sqrt(6.0/(input_size+output_size))
        return tf.random_uniform_initializer(-init_range,init_range)
    else:
        stddev= tf.sqrt(3.0/(input_size+output_size))
        return tf.truncated_normal_initializer(stddev=stddev)


# MNIST 데이터를 읽늗다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


learning_rate = 0.001
traing_approach = 25
batch_size = 100
default_node_size = 250
display_step = 1

dropout_rate = tf.placeholder("float")

X = tf.placeholder("float",[None,784])
Y = tf.placeholder("float",[None,10])

W1 = tf.get_variable("W1",shape=[784,default_node_size],initializer=xavier_init(784,default_node_size))
W2 = tf.get_variable("W2",shape=[default_node_size,default_node_size],initializer=xavier_init(default_node_size,default_node_size))
W3 = tf.get_variable("W3",shape=[default_node_size,default_node_size],initializer=xavier_init(default_node_size,default_node_size))
W4 = tf.get_variable("W4",shape=[default_node_size,default_node_size],initializer=xavier_init(default_node_size,default_node_size))
W5 = tf.get_variable("W5",shape=[default_node_size,default_node_size],initializer=xavier_init(default_node_size,default_node_size))
W6 = tf.get_variable("W6",shape=[default_node_size,default_node_size],initializer=xavier_init(default_node_size,default_node_size))
W7 = tf.get_variable("W7",shape=[default_node_size,10],initializer=xavier_init(default_node_size,10))

B1= tf.Variable(tf.random_normal([default_node_size]))
B2 = tf.Variable(tf.random_normal([default_node_size]))
B3 = tf.Variable(tf.random_normal([default_node_size]))
B4 = tf.Variable(tf.random_normal([default_node_size]))
B5 = tf.Variable(tf.random_normal([default_node_size]))
B6 = tf.Variable(tf.random_normal([default_node_size]))
B7 = tf.Variable(tf.random_normal([10]))


_L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),B1))
L1 = tf.nn.dropout(_L1,dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),B2))
L2 = tf.nn.dropout(_L2,dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2,W3),B3))
L3 = tf.nn.dropout(_L3,dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3,W4),B4))
L4 = tf.nn.dropout(_L4,dropout_rate)
_L5 = tf.nn.relu(tf.add(tf.matmul(L4,W5),B5))
L5 = tf.nn.dropout(_L5,dropout_rate)
_L6 = tf.nn.relu(tf.add(tf.matmul(L5,W6),B6))
L6 = tf.nn.dropout(_L6,dropout_rate)

hyp = tf.add(tf.matmul(L6,W7),B7)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hyp,Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    for eproch in range(traing_approach):
        avg_cost = 0
        total_branch = int(mnist.train.num_examples/batch_size)

        for i in range(total_branch):
            batch_xs, batch_ys =mnist.train.next_batch(batch_size)
            session.run(optimizer,feed_dict={X:batch_xs,Y:batch_ys,dropout_rate:0.7})
            avg_cost += session.run(cost, feed_dict={X:batch_xs,Y:batch_ys,dropout_rate:0.7})/total_branch

        if(eproch % display_step ==0):
            print "Eproch","%04d" % (eproch+1), "cost:","{:.9f}".format(avg_cost)


    print("optimization finished")

    #Test Model
    correct_prediction = tf.equal(tf.argmax(hyp,1),tf.argmax(Y,1))
    #accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print "accuracy", accuracy.eval({X:mnist.test.images,Y:mnist.test.labels,dropout_rate:1})



