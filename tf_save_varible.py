import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
weights = tf.Variable(tf.random_normal([784,200],stddev=0.35),name='weights')
biases = tf.Variable(tf.zeros([200]),name='biases')
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess,'mynet/test.ckpt')
    print('OK~')
