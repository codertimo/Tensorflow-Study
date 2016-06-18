#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#Binary Classification을 위한 Logistic Regression에 관한 TensorFlow예제입니다
#시그모이드 함수(Logistic Function)를 이용해서 H(x)를 정의하고 Cost함수를 정의함

#파일 첫줄에 "#"이 있는 것은 주석으로 처리하여 읽지 않음
xy = np.loadtxt('data.txt',unpack=True,dtype='float32')
#첫줄부터 맨마지막 전까지 (-1은 끝에서 부터 세는것)
x_data = xy[0:-1]
#y데이터는 맨 마지막 번째
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

print x_data
print y_data

#4개의 데이터
W = tf.Variable(tf.random_uniform([1,len(x_data)],-1.0,1.0))
h = tf.matmul(W,X)
hyp = tf.div(1.,1.+tf.exp(-h))

cost = -tf.reduce_mean( Y*tf.log(hyp) + (1-Y)*tf.log(1-hyp))

optimizer = tf.train.GradientDescentOptimizer(1e-2)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.initialize_all_variables())

for step in range(10000):
    session.run(train,feed_dict={X:x_data,Y:y_data})
    if(step%1000 ==0):
        print step, session.run(cost,feed_dict={X:x_data,Y:y_data}), session.run(W)

print "----------------에측----------------------"
print session.run(hyp,feed_dict={X:[[1],[2],[2]]}) >0.5
print session.run(hyp,feed_dict={X:[[1],[5],[5]]}) >0.5
print session.run(hyp,feed_dict={X:[[1,1],[4,3],[3,5]]}) >0.5