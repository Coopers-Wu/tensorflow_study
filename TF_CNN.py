# Cooper Wu'Code

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
#只是用来消去警告的

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict = {xs:v_xs, ys:v_ys, keep_prob:1})
    return result

#定义weight
def weight_variable(shape):
    #产生随机变量：从截断的正态分布中输出随机值。
    #stddev 是标准差的意思 还有其他的参数
    '''
        shape: 一维的张量，也是输出的张量。
        mean: 正态分布的均值。
        stddev: 正态分布的标准差。
        dtype: 输出的类型。
        seed: 一个整数，当设置之后，每次生成的随机数都一样。
        name: 操作的名字。
    '''
    initial = tf.truncated_normal(shape,stddev=0.1)
    return  tf.Variable(initial)

def bias_variable(shape):
    #定义bias为0.1，然后return。
    initial = tf.constant(0.1,shape=shape)
    return  tf.Variable(initial)

def conv2d(x,W):
    #x:图片的所有信息
    #tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    #strides是个一维向量，长度为4.2，3需要设置为x,y方向的步长，1，4为1.
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding= 'SAME')
    #[1,y_movement,y_movement,1]
    #padding有两种：same和


def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #[0],[3] must is 1

xs = tf.placeholder(tf.float32,[None, 784])
ys = tf.placeholder(tf.float32,[None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])
#tensor转化成图像信息
#channel is 1
#-1和none的意思是类似的，由实际情况决定的，是一个维度信息。

##conv1 layer
W_conv1 = weight_variable([5,5,1,32]) #patch 5x5 insize 1 depth。 outsize 32 高度
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1) #output 28*28*32
h_pool1 = max_pool_2x2(h_conv1)#output 14*14*32

W_conv2 = weight_variable([5,5,32,64]) #patch 5x5 insize 32是图像的厚度 outsize 64 高度
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2) #output 14*14*64
h_pool2 = max_pool_2x2(h_conv2)#output 7*7*64

W_fc1 = weight_variable([7*7*64,1024])
#转化成1024个神经元
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])#把数据给变平，输入
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob: 0.5})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

#sess.run(tf.global_variables_initializer())
save_path = saver.save(sess,"mynet/cnn.ckpt")
print('save to path:',save_path)
