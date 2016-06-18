#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#파일 첫줄에 "#"이 있는 것은 주석으로 처리하여 읽지 않음
xy = np.loadtxt('data.txt',unpack=True,dtype='float32')
#첫줄부터 맨마지막 전까지 (-1은 끝에서 부터 세는것)
x_data = xy[0:-1]
#y데이터는 맨 마지막 번째
y_data = xy[-1]

# x_data = [[1.,0.,3.,0.,5.],[0.,2.,0.,4.,0.]]
# y_data = [1,2,3,4,5]

print x_data
print y_data

#4개의 데이터
W = tf.Variable(tf.random_uniform([1,len(x_data)],-5.0,5.0))

hyp = tf.matmul(W,x_data)

cost = tf.reduce_mean(tf.square(hyp-y_data))

optimizer = tf.train.GradientDescentOptimizer(1e-2)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.initialize_all_variables())

for step in range(5000):
    session.run(train)
    if(step%100 ==0):
        print step, session.run(cost), session.run(W)
