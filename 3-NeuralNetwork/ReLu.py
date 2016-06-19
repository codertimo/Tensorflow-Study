#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

## XORCalculation을 Neural Net을 이용하여 구하는 스크립트

#파일 첫줄에 "#"이 있는 것은 주석으로 처리하여 읽지 않음
xy = np.loadtxt('xor_data.txt',unpack=True,dtype='float32')
#첫줄부터 맨마지막 전까지 (-1은 끝에서 부터 세는것)
x_data = np.transpose(xy[0:-1])
#y데이터는 맨 마지막 번째
y_data = np.reshape(xy[-1], (len(xy[-1]), 1))

print x_data
print y_data

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 변수들에 대한 설명은 https://youtu.be/9i7FBbcZPMA?t=7m13s
# 첫번째 레이어에서 2개의 노드가 있고, 각각은 W2값을 2개 갖기 때문에, 2x2 4개이다[a,b]가 노드 갯수, b가 W갯수
#3200 0.0960798 : second Layer가 10개였을 때
#3200 0.693189 :second Layer가 2개였을 때

#1st Layer : 2개 입력 10개로 출력
W1 = tf.Variable(tf.random_uniform([2,1000],-1.0,1.0))
#Secode Layer : 10개 입력 1개로 출력
W2 = tf.Variable(tf.random_uniform([1000,1],-1.0,1.0))


#첫번째 레이어의 b인 b1은 노드 2개의 바이어스를 갖음[b]노드 갯수
b1 = tf.Variable(tf.zeros([1000]),name="Bias1")
#두번쩨 레이어에서는 노드가 하나임으로 바이어스도 1개
b2 = tf.Variable(tf.zeros([1]),name="Bias2")

#첫번째 레이어(2개의 노드)에서 각각 시그모이드를 한후에 행렬형태로 출력한다L2[node1,node2]
L2 = tf.nn.relu(tf.matmul(X,W1)+b1)
#입력받은 두 데이터를 바탕으로 시그모이드를 다시 실행한다 = 가정
hyp = tf.sigmoid(tf.matmul(L2,W2)+b2)

#Logistic Regression Cost Function
cost = -tf.reduce_mean(Y*tf.log(hyp) + (1-Y)*tf.log(1-hyp))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.initialize_all_variables())

for step in range(10000):
    session.run(train, feed_dict={X:x_data, Y:y_data})
    if(step%200==0):
        print step, session.run(cost, feed_dict={X:x_data,Y:y_data})


correct_predection = tf.equal(tf.floor(hyp+0.5),Y)
accuracy = tf.reduce_mean(tf.cast(correct_predection,"float"))
print session.run([hyp, tf.floor(hyp+0.5),correct_predection,accuracy],feed_dict={X:x_data,Y:y_data})
