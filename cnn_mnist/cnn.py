import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import preprocessing
import pylab
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

mnist=input_data.read_data_sets("./MNIST_data",one_hot=True)
print(mnist.train.num_examples)

# 定义初始权重函数
def init_weights(shape):
    # 服从截断的正态分布， 方差为0.1
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial_value=initial)

# 定义偏置的初始值
def init_bias(shape):
    #initial = tf.constant(value=0.1, shape=shape)
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

# 定义一个2D的单通道卷积层
def conv2d(x, W):
    # stage=1,卷积仅仅导致边缘损失padding=“same”代表用0来填充保证输入和输出的图片像素一致
    # 2d to 4d
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding="VALID")

# 定义pool函数，maxpool用来降为提取显著的特征
def max_pool_2x2(x):
    # 两个维度采取边缘填充0，stride=2：那么图像维度行列都变为原来的1/2
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# 样本定义
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
# 真实值
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# 要把数据转化为28x28的图片
x_image = tf.reshape(x, [-1, 28, 28, 1])  # -1:类似于numpy，后面的1：代表通道

# 定义第一个卷积层, 定义32个单通道5x5的核，提取32个特征
W_conv1 = init_weights(shape=[13, 13, 1, 16])
b_conv1 = init_bias(shape=[16])
# 第一层卷积层的输出
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 图像变成14x14 x 32

# 定义第二层卷积层
W_conv2 = init_weights(shape=[3, 3, 16, 32])  # 单通道变成32，提起64个特征，定义5x5核
b_conv2 = init_bias(shape=[32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)   # 7x7 x 64

# h_pool2转化为1维度的数据
num = 3 * 3 * 32
h_pool2_flat = tf.reshape(h_pool2, [-1, num])

# 隐藏节点设定为1024:全连接层
W_fc1 = init_weights([num, 512])
b_fc1 = init_weights([512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(x=h_fc1, keep_prob=keep_prob)

# 全连接输出层
W_fc2 = init_weights([512, 10])
b_fc2 = init_weights([10])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y = tf.nn.softmax(h_fc2)

# loss function
loss = tf.reduce_mean(
    - tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])  # 先计算求和每行的交叉熵，然后去每个样本的交叉熵均值
)
with tf.Session() as sess:
    # train
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.global_variables_initializer().run()  # 图的所有变量初始化

    # 训练模型
    for i in range(550*30):
        batchX, batchY = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.7})
        # print(h_pool2_flat.shape)
        if i % 550 == 0:
            print(sess.run(loss, feed_dict={x: batchX, y_: batchY, keep_prob: 0.7}))

    # test
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    output=tf.argmax(y,1)
    batch_xs, batch_ys = mnist.test.next_batch(4)
    outputval, predv = sess.run([output, y], feed_dict={x: batch_xs, keep_prob: 1.0})
    print(outputval, predv, batch_ys)
'''
    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
'''

