#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

X = None

# [a,b,c,d]
# a,b 필터의 크기 / c : 필터의 depth / d : 필터의 갯수(Ouput Layer)
W= tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))

#stride [1,a,b,1] a,b 가로세로로 몇칸씩 움직일까
tf.nn.conv2d(X,W,strides=[1,1,1,1],padding='SAME')
#-> Output [1,1,1]짜리 크기의 레이어가 총 32개 만들어짐

#ReLU까지 적용한 Layer
c1 = tf.nn.relu(tf.nn.conv2d(X,W,strides=[1,1,1,1],padding='SAME'))

#max_pool filter 함수
#ksize = [1,a,b,1] a,b 크기에서 최대값을 찾기!
#stride = [1,a,b,1] 한번에 a,b만큼 움직인다
tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#-> 이미지의 사이즈를 1/2로 줄임


#---------------------Example for calculating size of shape-------------------

def init_weight(size_array, stddev=0.01):
    return tf.Variable(tf.random_normal(size_array,stddev=stddev))

droprate_hidden  = tf.placeholder("float")
droprate_last  = tf.placeholder("float")

w1 = init_weight([28,28,1,32]) # 3x3x1 Input, 32 Ouput
w2 = init_weight([14,14,32,64]) # 3x3x32 Input, 64 Ouput
w3 = init_weight([7,7,64,128]) # 3x3x64 Input, 128 Ouput

# X : (?,28,28,1)
c1 = tf.nn.relu(tf.nn.conv2d(X,w1,strides=[1,1,1,1],padding='SAME'))  ## c1 :(?,28,28,32) -> 28x28 32 Ouput
l1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') ## l1 : (?,14,14,32) -> 14x14 32 Ouput
l1 = tf.nn.dropout(l1,droprate_hidden)

c2 = tf.nn.relu(tf.nn.conv2d(l1,w2,strides=[1,1,1,1],padding='SAME'))  ## c1 :(?,14,14,64) -> 14x14 64 ouput
l2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') ## l1 : (?,7,7,64) ->7x7 64 Ouput
l2 = tf.nn.dropout(l2,droprate_hidden)

c3 = tf.nn.relu(tf.nn.conv2d(l2,w3,strides=[1,1,1,1],padding='SAME'))  ## c1 :(?,7,7,128) -> 14x14 128 ouput
l3 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') ## l1 : (?,4,4,128) ->7x7 128 Ouput
l3 = tf.nn.dropout(l3,droprate_hidden)

#4*4*128개의 input , 625개의 Ouput
w4 = init_weight([4*4*128,625])
#최종 Weight, 625개 input, 10개 Ouput
w_h = init_weight([625,10])

l3 = tf.reshape(l3,[-1,w4.get_shape().as_list()[0]])
l3 = tf.nn.dropout(l3,droprate_hidden)

l4 = tf.nn.relu(tf.matmul(l3,w4))
l4 = tf.nn.dropout(l4,droprate_last)

hyp = tf.matmul(l4,w_h)

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(hyp,Y))
optimizer =tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
predict_optimizer = tf.arg_max(hyp,1)

with tf.Session() as session:
    for step in range(100):
        for start,end in zip(range(0,len(x_data),128),range(128,len(y_data),128)):
            session.run(optimizer,feed_dict={X:x_data[start:end],Y:y_data[start:end],droprate_hidden:0.8,droprate_last:0.5})

        test_index = np.arange(len(test_x)) #[1,2,3,..]
        np.random.suffle(test) #[7,13,1,9...]
        test_index = test[0:256] # 그중에서 256개 가져오기
        accruacy = np.mean(np.argmax(test_y[test_index],axis=1)) == \
                   session.run(predict_optimizer,feed_dict={X:test_x[test_index],Y:test_y[test_index],droprate_last:1.0,droprate_hidden:1.0})

        print step, accruacy


