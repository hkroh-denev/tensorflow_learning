"""
    DCGAN implementation by hkroh
    July 5, 2017
"""

import numpy as np
import tensorflow as tf
import os
import mnist_data
import image_util

def conv2d(input_v, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_v.get_shape()[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_v, w, strides=[1, d_h, d_w, 1], padding="SAME")
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(conv, biases)

def deconv2d(input_v, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_v.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_v, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv

def fully_connected(input_v, output_size, name="fc", stddev=0.02):
    shape = input_v.get_shape().as_list()
    with tf.variable_scope(name or "fully_connected"):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_v, w) + b

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name or "lrelu"):
        return tf.maximum(x, leak*x)

def batch_norm(input_v, train=True, name="bn"):
    return tf.contrib.layers.batch_norm(input_v, decay=0.9, updates_collections=None,
                    epsilon=1e-5, scale=True, is_training=train, scope=name)


class DCGAN():
    def __init__(self, batch_size=64, channel=1):

        # hyper parameters
        self.batch_size = batch_size
        self.max_filter = 160
        self.input_channel = channel

        # placeholder
        self.input_noise = tf.placeholder(shape=[self.batch_size, 100], dtype=tf.float32, name="input")

        with tf.variable_scope("Generator"):
            self.L0 = fully_connected(self.input_noise, 7 * 7 * self.max_filter, "linear0")
            self.L0 = tf.reshape(self.L0, [self.batch_size, 7, 7, self.max_filter], name="reshape")
            #self.L0 = batch_norm(self.L0, name="batch_norm0")
            #self.L0 = tf.nn.relu(self.L0, name="relu0")

            self.L1 = deconv2d(self.L0, [self.batch_size, 14, 14, self.max_filter//2], name="deconv1")
            self.L1 = batch_norm(self.L1, name="batch_norm1")
            self.L1 = tf.nn.relu(self.L1, name="relu1")

            self.L2 = deconv2d(self.L1, [self.batch_size, 14, 14, self.max_filter//2], 5, 5, 1, 1, name="deconv2")
            self.L2 = batch_norm(self.L2, name="batch_norm2")
            self.L2 = tf.nn.relu(self.L2, name="relu2")

            self.L3 = deconv2d(self.L2, [self.batch_size, 28, 28, self.input_channel], name="deconv3")
            self.generated = tf.nn.tanh(self.L3)

        with tf.variable_scope("Discriminator"):
            self.inputD = self.generated
            self.labelD = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.float32, name="label")

            self.L4 = conv2d(self.inputD, self.max_filter//2, name="conv4")
            self.L4 = batch_norm(self.L4, name="batch_norm4")
            self.L4 = lrelu(self.L4, name="lrelu4")

            self.L5 = conv2d(self.L4, self.max_filter//2, 5, 5, 1, 1, name="conv5")
            self.L5 = batch_norm(self.L5, name="batch_norm5")
            self.L5 = lrelu(self.L5, name="lrelu5")

            self.L6 = conv2d(self.L5, self.max_filter, name="conv6")
            self.L6 = batch_norm(self.L6, name="batch_norm6")
            self.L6 = lrelu(self.L6, name="lrelu6")

            self.output = fully_connected(tf.reshape(self.L6, [self.batch_size, -1]), 1, "Linear_Out")
            self.predict = tf.nn.sigmoid(self.output, "Output")

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.labelD))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        print('d_vars', [var.name for var in self.d_vars])
        self.d_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss, var_list=self.d_vars)
        self.g_vars = [var for var in t_vars if 'Generator' in var.name]
        self.g_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss, var_list=self.g_vars)

        self.loss_sum = tf.summary.scalar('loss', self.loss)
        self.sum = tf.summary.merge([self.loss_sum])

#
#   hyper-parameter
#
batch_size = 200
dcgan = DCGAN(batch_size=batch_size, channel=1)

# load MNIST data
images, labels = mnist_data.load_mnist('./mnist')
input_img = []
for i in range(60000):
    input_img.append(images[i])
input_img = np.array(input_img)
image_util.save_images('output.png', input_img[0:64], [8,8])

# preprocess images
input_img = input_img / 127.0 - 1.0
input_img = np.reshape(input_img, [60000, 28, 28, 1])

# init TF
init = tf.global_variables_initializer()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs_mnist', sess.graph)

    # Begin training
    epoch = 0
    batch_count = 60000 // batch_size
    while epoch < 500:

        for i in range(batch_count):

            z = np.random.normal(0.0, 1.0, size=[batch_size, 100])

            batch_input = input_img[i*batch_size:(i+1)*batch_size]

            label_real = np.ones([batch_size, 1], np.float32)
            label_real -= 0.1
            label_fake = np.zeros([batch_size, 1], np.float32)
            label_fake += 0.1

            _, d_loss1, ds = sess.run([dcgan.d_optim, dcgan.loss, dcgan.sum],
                feed_dict={dcgan.input_noise:z, dcgan.labelD: label_fake})

            _, d_loss2, ds = sess.run([dcgan.d_optim, dcgan.loss, dcgan.sum],
                feed_dict={dcgan.inputD:batch_input, dcgan.labelD: label_real})

            _, g_loss1, gen, gs = sess.run([dcgan.g_optim, dcgan.loss, dcgan.generated, dcgan.sum],
                feed_dict={dcgan.input_noise:z, dcgan.labelD: label_real})

            _, g_loss2, gen, gs = sess.run([dcgan.g_optim, dcgan.loss, dcgan.generated, dcgan.sum],
                feed_dict={dcgan.input_noise:z, dcgan.labelD: label_real})

            d_loss = d_loss1 + d_loss2
            g_loss = g_loss1 + g_loss2

            writer.add_summary(ds, epoch * batch_count + i)
            writer.add_summary(gs, epoch * batch_count + i)

            print('EPOCH {}'.format(epoch), '[{}/{}] D_Loss: {}, G_Loss: {}'.format(i, batch_count, d_loss, g_loss))

            if i % 50 == 49:
                # de-preprocess image & save
                gen = np.reshape(gen, [batch_size, 28, 28])
                gen_img = gen + 1.0 * 127.0
                if not os.path.exists('gen_mnist'):
                    os.makedirs('gen_mnist')
                image_util.save_images('gen_mnist/{}_{}.png'.format(epoch, i), gen_img[0:64], [8,8])

        epoch += 1
