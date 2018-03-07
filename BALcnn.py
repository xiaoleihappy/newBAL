import tensorflow as tf
import numpy as np
import os
import math
#import matplotlib.pyplot as plt
import astropy.io.fits as pyfits 
import scipy 

def pre_dis(x,k):
     m=float(len(x))
     average=x.sum()/m
     testy=x
     c=[j for j,s in enumerate(testy) if abs(s-average)>k]
     testy=np.delete(testy,c) 
     y=(x-min(testy))/(max(testy)-min(testy))
     return y

def read_data(x,y,n,m):
    global batch_xs,batch_ys,testxs,testys
    batch_xs= np.zeros( (n,4550) )
    batch_ys= np.zeros( (n,2) )
    testxs= np.zeros( (m,4550) )
    testys= np.zeros( (m,2) )
    a=[]
    b=[]
   # batch_xs[0:int(n/2)]=pre_dis(x[0:int(n/2)][1][20:4570],5)
   # batch_xs[int(n/2):n]=pre_dis(y[0:int(n/2)][1][20:4570],5)
   # batch_ys[0:int(n/2)]=[[1,0] for i in range (int(n/2))]
   # batch_ys[int(n/2):n]=[[0,1] for i in range (int(n/2))]
   # testxs[0:int(m/2)]=pre_dis(x[0:int(n/2+1000)][1][20:4570],5)
   # testys[0:int(m/2)]=[[1,0] for i in range (int(n/2))]
   # testxs[int(m/2):m]=pre_dis(y[0:int(n/2+1000)][1][20:4570],5)
   # testys[int(m/2):m]=[[0,1] for i in range (int(n/2))]

    for i in range (int(n/2)):
        batch_xs[i]=x[i][1][20:4570]
        #batch_xs [i] = pre_dis(a,5)
        batch_ys [i] = 1
    for i in range(int(n/2)):
        batch_xs[i+int(n/2)]=y[i][1][20:4570]
        #batch_xs[i+int(n/2)]=pre_dis(b,5)
        batch_ys[i+int(n/2)]=0
    for i in range(m):
        testxs[i]=x[i+1000][1][20:4570]
        # = pre_dis(a,5)
        testys[i]= [1,0]
     
   # for i in range(m):
   #     b=y[i+1000][1][20:4570]
   #     testxs[i] = pre_dis(b,5)
   #     testys[i]= [0,1]
   # print (testxs[0][4:20])
   
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x1(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')
def max_pool_5x1(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,1,5,1], strides=[1,1,5,1], padding='SAME')

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

bal=pyfits.open("DR12_BAL.fits")
baldata=bal[1].data
nobal=pyfits.open("DR12_noBAL.fits")
nobaldata=nobal[1].data
read_data(baldata,nobaldata,200,20)

# define placeholder for inputs to network

xs = tf.placeholder(tf.float32, [None, 4550])   # 4550*1
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 1,4550, 1])
#data input#

#tf structure start#
#conv1#

W_conv1 = weight_variable([1,32,1,300]) # 2  in size 1, out size 5
b_conv1 = bias_variable([300])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)# output size 4550*1*5
h_pool1 = max_pool_2x1(h_conv1) # output size 2275*1*5
#h_pool1=h_conv1

#conv2#

W_conv2 = weight_variable([1,32,300,500]) # patch 5, in size 5, out size 25
b_conv2 = bias_variable([500])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 2275*1*5
h_pool2 = max_pool_5x1(h_conv2)                                    # output size 455*1*25
#h_pool2=h_conv2

## fc1 layer ##
W_fc1 = weight_variable([500, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 455, 1, 25] ->> [n_samples, 455*1*25
h_pool2_flat = tf.reshape(h_pool2, [-1,455*500*1 ])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


##fc2 layer##
W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# the error between prediction and real data
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                              reduction_indices=[1]))       # loss

loss=tf.sigmoid(prediction)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess = tf.Session()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
#print(h_pool2.shape)
#print (h_pool2_flat.shape)
for i in range(1000):
    
    sess.run(train_step, feed_dict={xs:   batch_xs, ys:   batch_ys, keep_prob: 0.5}) 
    if i % 50 == 0:
        print(compute_accuracy(testxs,testys))
