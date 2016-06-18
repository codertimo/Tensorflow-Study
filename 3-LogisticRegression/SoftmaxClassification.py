#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#파일 첫줄에 "#"이 있는 것은 주석으로 처리하여 읽지 않음
xy = np.loadtxt('softmax.txt',unpack=True,dtype='float32')
#첫줄부터 맨마지막 전까지 (-1은 끝에서 부터 세는것)
x_data = np.transpose(xy[0:3])
#y데이터는 맨 마지막 번째
y_data = np.transpose(xy[3:])
print x_data
print y_data

X = tf.placeholder("float",[None,3]) # x1,x2, and bias(None을 넣은 이유 : 데이터가 몇개인지 모르기 때문)
Y = tf.placeholder("float",[None,3]) # A,B,C
W = tf.Variable(tf.zeros([3,3]))

hyp = tf.nn.softmax(tf.matmul(X,W))

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hyp),reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

session = tf.Session()
session.run(tf.initialize_all_variables())

for step in range(10000):
    session.run(optimizer,feed_dict={X:x_data,Y:y_data})
    if(step%1000==0):
        print step, session.run(cost, feed_dict={X:x_data,Y:y_data}), session.run(W)


# 0: A, 1:B, 2:C

a = session.run(hyp,feed_dict={X:[[1,11,7]]})
print a, session.run(tf.arg_max(a,1))

b = session.run(hyp,feed_dict={X:[[1,3,4]]})
print b,session.run(tf.arg_max(b,1))

c = session.run(hyp,feed_dict={X:[[1,1,0]]})
print c, session.run(tf.arg_max(c,1))

all = session.run(hyp,feed_dict={X:[[1,11,7],[1,3,4],[1,1,0]]})
print all, session.run(tf.arg_max(all,1))