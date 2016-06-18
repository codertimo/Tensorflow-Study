#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#파일 첫줄에 "#"이 있는 것은 주석으로 처리하여 읽지 않음
import numpy as np
xy = np.loadtxt('data.txt',unpack=True,dtype='float32')
#첫줄부터 맨마지막 전까지 (-1은 끝에서 부터 세는것)
x_data = xy[0:-1]
#y데이터는 맨 마지막 번째
y_dya = xy[-1]