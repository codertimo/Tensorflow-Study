#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
# MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
x_data, y_data, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


print len(x_data)

def init_weight(size_array, name,stddev=0.01):
    return tf.Variable(tf.random_normal(size_array,stddev=stddev),name=name)


def model(X,w1,w2,w3,w4,w_h,hidden_rate,last_rate):
    # X : (?,28,28,1)
    c1 = tf.nn.relu(tf.nn.conv2d(X,w1,strides=[1,1,1,1],padding='SAME'))  ## c1 :(?,28,28,32) -> 28x28 32 Ouput
    l1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') ## l1 : (?,14,14,32) -> 14x14 32 Ouput
    l1 = tf.nn.dropout(l1,hidden_rate)

    c2 = tf.nn.relu(tf.nn.conv2d(l1,w2,strides=[1,1,1,1],padding='SAME'))  ## c1 :(?,14,14,64) -> 14x14 64 ouput
    l2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') ## l1 : (?,7,7,64) ->7x7 64 Ouput
    l2 = tf.nn.dropout(l2,hidden_rate)

    c3 = tf.nn.relu(tf.nn.conv2d(l2,w3,strides=[1,1,1,1],padding='SAME'))  ## c1 :(?,7,7,128) -> 14x14 128 ouput
    l3 = tf.nn.max_pool(c3, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') ## l1 : (?,4,4,128) ->7x7 128 Ouput
    l3 = tf.reshape(l3,[-1,w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3,hidden_rate)

    l4 = tf.nn.relu(tf.matmul(l3,w4))
    l4 = tf.nn.dropout(l4,last_rate)

    hyp = tf.matmul(l4,w_h)
    return hyp

##---------- 1.Initialize ---------

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])
droprate_hidden  = tf.placeholder("float")
droprate_last  = tf.placeholder("float")

x_data = x_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
test_x = test_x.reshape(-1, 28, 28, 1)  # 28x28x1 input img

w1 = init_weight([28,28,1,32],"w1") # 3x3x1 Input, 32 Ouput
w2 = init_weight([14,14,32,64],"w2") # 3x3x32 Input, 64 Ouput
w3 = init_weight([7,7,64,128],"w3") # 3x3x64 Input, 128 Ouput
w4 = init_weight([4*4*128,625],"w4")
w_h = init_weight([625,10],"wh")

batch_size = 128
test_size = 256

#-------- 2. run train ----------

hyp = model(X,w1,w2,w3,w4,w_h,droprate_hidden,droprate_last)
print hyp
print Y
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hyp,Y))

tf.train.AdamOptimizer
optimizer =tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
predict_optimizer = tf.arg_max(hyp,1)

with tf.Session() as session:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    try:
        saver.restore(session, "./model_1.ckpt")
    except:
        pass

    for step in range(100):

        test_index = np.arange(len(test_x)) #[1,2,3,..]
        np.random.shuffle(test_index)
        test_index = test_index[0:256] # 그중에서 256개 가져오기

        count_size = len(x_data)/batch_size
        count = 0
        avg_time = 0.0
        for start,end in zip(range(0,len(x_data),batch_size),range(batch_size,len(y_data),batch_size)):
            if(count%50==0):
                test_index = np.arange(len(test_x)) #[1,2,3,..]
                np.random.shuffle(test_index)
                test_index = test_index[0:256] # 그중에서 256개 가져오기

                accruacy = np.mean(np.argmax(test_y[test_index],axis=1) \
                   == session.run(predict_optimizer,feed_dict={X:test_x[test_index],Y:test_y[test_index],droprate_last:1.0,droprate_hidden:1.0}))
                print "Accuracy :"+str(accruacy)
                save_path = saver.save(session, "./model_1.ckpt")
                print("모델들이 저장되었습니다")
            count+=1
            start_time = time.time()
            session.run(optimizer,feed_dict={X:x_data[start:end],Y:y_data[start:end],droprate_hidden:0.8,droprate_last:0.5})
            avg_time+= time.time()-start_time
            print str(count)+"/"+str(count_size),start,end,("예상소요시간:"+str(int(round(avg_time/count*(count_size-count))/60))+"분"),("cost:"+str(session.run(cost,feed_dict={X:x_data[start:end],Y:y_data[start:end],droprate_hidden:0.8,droprate_last:0.5})))


#-------------------------------