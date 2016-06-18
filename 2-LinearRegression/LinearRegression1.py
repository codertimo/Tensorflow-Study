#-*- coding: utf-8 -*-
import tensorflow as tf

#값 설정
x_data = [1,2,3]
y_data = [1,2,3]

# -1.0 부터 1.0까지의 수중 랜덤하게 갖는 변수를 정의함
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

#1차 방정식 ax+b의 식을 세움
hyp = W * x_data + b

# 손실함수는 (가정 - 실제)^2 의 평균으로 연산을 정의함
cost = tf.reduce_mean(tf.square( hyp - y_data ))

#손실함수를 최소로 만들도록 정의함
optimizer = tf.train.GradientDescentOptimizer(1e-2)
train = optimizer.minimize(cost)

#W,B의 변수들을 초기화 시켜주는 연산을 정의함
init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

for step in range(2001):
    session.run(train)
    if(step%20 == 0):
        print step,session.run(cost),session.run(W),session.run(b)




