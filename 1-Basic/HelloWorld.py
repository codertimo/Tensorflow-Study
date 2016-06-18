#-*- coding: utf-8 -*-
import tensorflow as tf

# tf.constant는 operation 즉 하나의 연산작업
hello = tf.constant("Hi TensorFlow")

# print해보면 결과 : Tensor("Const:0", shape=(), dtype=string)
# 즉 상수가 아닌 하나의 연산 작업
print hello

session = tf.Session()
print session.run(hello)

#연산작업 2

a = tf.constant(1)
b = tf.constant(2)

c= a+b

print session.run(c)