#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#Binary Classification을 위한 Logistic Regression에 관한 TensorFlow예제입니다
#시그모이드 함수(Logistic Function)를 이용해서 H(x)를 정의하고 Cost함수를 정의함

#파일 첫줄에 "#"이 있는 것은 주석으로 처리하여 읽지 않음
xy = np.loadtxt('xor_data.txt',unpack=True,dtype='float32')
#첫줄부터 맨마지막 전까지 (-1은 끝에서 부터 세는것)
x_data = xy[0:-1]
#y데이터는 맨 마지막 번째
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1,len(x_data)],-1.0,1.0))

# Y = WX
h = tf.matmul(W,X)
#SigMoid Function 1/1+e^(-WX)
hyp = tf.div(1., 1+tf.exp(-h))
#Logistic Regression Cost Function
cost = -tf.reduce_mean(Y*tf.log(hyp) + (1-Y)*tf.log(1-hyp))

optimizer = tf.train.GradientDescentOptimizer(1e-2)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.initialize_all_variables())

for step in range(10000):
    session.run(train, feed_dict={X:x_data, Y:y_data})
    if(step%200==0):
        print step, session.run(cost, feed_dict={X:x_data,Y:y_data}), session.run(W)


correct_predection = tf.equal(tf.floor(hyp+0.5),Y)
accuracy = tf.reduce_mean(tf.cast(correct_predection,"float"))
print session.run([hyp, tf.floor(hyp+0.5),correct_predection,accuracy],feed_dict={X:x_data,Y:y_data})

##정확도가 0.5밖에 되지 않음(일반적 Training) -> NN으로 해보자