# #-*- coding: utf-8 -*-
# import tensorflow as tf
# import numpy as np
#
# #Tip For Neural Network Training Process
#
# #-----------Weight Value initializing----------
#
# #1. Weight Value Initializing <Regulation Strength>
#
# W = np.random.randn(input_lenth,output_lenth)/np.sqrt(input_lenth/2)
#
# ###--------Prevent Over Fitting-----------------
#
# # Cost + Under Code
# 12reg = 0.001 * tf.reduce_sum(tf.square(W))
#
# #2. Drop Out
# droprate  = tf.placeholder("float")
# _L1 = tf.nn.relu(tf.add(tf.matmul(X,W),B))
# L1 = tf.nn.dropout(_L1,droprate)
#
# #When Train
# session.run(optimizer,feed_dict={X:x_data,Y:y_data,droprate:0.7})
#
# #When Evaluation
# accruacy.eval({X:x_data, Y:y_data, droprate:1})
#
# #----------------------------------------------------
#
