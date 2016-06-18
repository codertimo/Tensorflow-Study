#-*- coding: utf-8 -*-

# 함수를 정의할때에 f(a,b)로 정의한다
# 텐서플로우는 연산작업을 변수로 받기 때문에 함수를 정의할때 임의의 변수가 필요하다

import tensorflow as tf

a = tf.placeholder(tf.int64)
b = tf.placeholder(tf.int64)

#값을 갖고있진 않지만 변수로 지정함

add = tf.add(a,b)
mul = tf.mul(a,b)


with tf.Session() as session:
    print "add %i" % session.run(add,feed_dict={a:1, b:2})
    print "mul %i" % session.run(mul,feed_dict={a:20, b:3})