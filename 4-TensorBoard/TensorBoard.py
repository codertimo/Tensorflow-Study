#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#1. 변수 선언법

X = tf.placeholder(tf.float32,name="X-input")
Y = tf.placeholder(tf.float32,name="Y-input")

W1 = tf.Variable(tf.random_uniform([2,1000],-1.0,1.0),name="Weight1")
W2 = tf.Variable(tf.random_uniform([1000,1],-1.0,1.0), name="Weight2")

b1 = tf.Variable(tf.zeros([1000]),name="Bias1")
b2 = tf.Variable(tf.zeros([1]),name="Bias2")

#2. Graph View를 위한 Scope 작성법

with tf.name_scope("Layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X,W1)+b1)

with tf.name_scope("Layer3") as scope:
    hyp = tf.sigmoid(tf.matmul(L2,W2)+b2)

with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y*tf.log(hyp) + (1-Y)*tf.log(1-hyp))
    cost_summary = tf.scalar_summary("cost",cost)

with tf.name_scope("opetimizer") as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(cost)

#3. 히스토그램 추가하기

h1 = tf.histogram_summary("Weight1",W1)
h2 = tf.histogram_summary("Weight2",W2)

b1_historgram = tf.histogram_summary("Bias1",b1)
b2_histogram = tf.histogram_summary("Bias2",b2)

#4. 위에서 생성한 Histogram과 변수들을 통합

merged = tf.merge_all_summaries()

#5. (Log) Writer 만들기

session = tf.Session()
#1. 로그 파일에 저장
writer = tf.train.SummaryWriter("../log/tensorboard-log1",session.graph_def)

for step in range(2000):
    if(step%100 ==0):
        #2. Summary 통합 연산
        summary = session.run(merged)
        #3. writer에 summary(log데이터) 추가
        writer.add_summary(summary)


#6. Launch TensorBoard
# command line : tensorboard --logdir=../log/tensorboard-log1
