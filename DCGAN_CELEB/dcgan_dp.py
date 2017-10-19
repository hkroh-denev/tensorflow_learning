"""
    DCGAN implementation by hkroh
    July 5, 2017

    data pipelining version
"""

import os
import numpy as np
import tensorflow as tf
import image_util
import data_queue

def conv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding="SAME")
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(conv, biases)

def deconv2d(x, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv

def fully_connected(x, output_size, name="fc", stddev=0.02):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name or "fully_connected"):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name or "lrelu"):
        return tf.maximum(x, leak*x)

def batch_norm(x, train=True, name="bn"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None,
                    epsilon=1e-5, scale=True, is_training=train, scope=name)

class DCGAN():
    def __init__(self, batch_size=64, channel=3, input_pipeline=None):

        # hyper parameters
        self.batch_size = batch_size
        self.max_filter = 512
        self.input_channel = channel
        self.input_pipeline = input_pipeline

        # placeholder
        self.input_noise = tf.placeholder(shape=[self.batch_size, 100], dtype=tf.float32, name="input")
        self.is_real =tf.placeholder(dtype=tf.bool, name="is_real")

        with tf.variable_scope("Generator"):
            self.L0 = fully_connected(self.input_noise, 4 * 4 * self.max_filter, "linear0")
            self.L0 = tf.reshape(self.L0, [self.batch_size, 4, 4, self.max_filter], name="reshape")
            self.L0 = batch_norm(self.L0, name="batch_norm0")
            self.L0 = tf.nn.relu(self.L0, name="relu0")

            self.L1 = deconv2d(self.L0, [self.batch_size, 8, 8, self.max_filter//2],  name="deconv1")
            self.L1 = batch_norm(self.L1, name="batch_norm1")
            self.L1 = tf.nn.relu(self.L1, name="relu1")

            self.L2 = deconv2d(self.L1, [self.batch_size, 16, 16, self.max_filter//4], name="deconv2")
            self.L2 = batch_norm(self.L2, name="batch_norm2")
            self.L2 = tf.nn.relu(self.L2, name="relu2")

            self.L3 = deconv2d(self.L2, [self.batch_size, 32, 32, self.max_filter//8], name="deconv3")
            self.L3 = batch_norm(self.L3, name="batch_norm3")
            self.L3 = tf.nn.relu(self.L3, name="relu3")

            self.L4 = deconv2d(self.L3, [self.batch_size, 64, 64, self.input_channel], name="deconv4")
            self.generated = tf.nn.tanh(self.L4)

        with tf.variable_scope("Discriminator"):
            self.inputD = tf.where(self.is_real, input_pipeline, self.generated)
            #self.inputD = tf.cond(self.is_real, lambda: input_pipeline, lambda: self.generated)
            self.input_label = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.float32, name="input_label")

            self.L5 = conv2d(self.inputD, self.max_filter//16, name="conv5")
            self.L5 = batch_norm(self.L5, name="batch_norm5")
            self.L5 = lrelu(self.L5, name="lrelu5")

            self.L6 = conv2d(self.L5, self.max_filter//8, name="conv6")
            self.L6 = batch_norm(self.L6, name="batch_norm6")
            self.L6 = lrelu(self.L6, name="lrelu6")

            self.L7 = conv2d(self.L6, self.max_filter//4, name="conv7")
            self.L7 = batch_norm(self.L7,  name="batch_norm7")
            self.L7 = lrelu(self.L7, name="lrelu7")

            self.L8 = conv2d(self.L7, self.max_filter//2, name="conv8")
            self.L8 = batch_norm(self.L8, name="batch_norm8")
            self.L8 = lrelu(self.L8, name="lrelu8")

            self.L9 = fully_connected(tf.reshape(self.L8, [self.batch_size, -1]), 1, "linear9")
            self.output = tf.nn.sigmoid(self.L9, "output")

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.L9, labels=self.input_label))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        self.d_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, epsilon=0.001).minimize(self.loss, var_list=self.d_vars)
        self.g_vars = [var for var in t_vars if 'Generator' in var.name]
        self.g_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, epsilon=0.001).minimize(self.loss, var_list=self.g_vars)

        self.d_loss_sum = tf.summary.scalar('d_loss', self.loss)
        self.g_loss_sum = tf.summary.scalar('g_loss', self.loss)
        self.d_sum = tf.summary.merge([self.d_loss_sum])
        self.g_sum = tf.summary.merge([self.g_loss_sum])


# training parameter
max_epoch = 1000
save_image_interval = 200
label_smoothing = 0.1

# model parameters
batch_size = 100
channel_num = 3
data_count = 350000

datafiles = [
    './celeb_tfrec/celeb_train_00000-of-00005.tfrecord',
    './celeb_tfrec/celeb_train_00001-of-00005.tfrecord',
    './celeb_tfrec/celeb_train_00002-of-00005.tfrecord',
    './celeb_tfrec/celeb_train_00003-of-00005.tfrecord',
    './celeb_tfrec/celeb_train_00004-of-00005.tfrecord'
]

data_pipeline = data_queue.make_data_pipeline(datafiles, num_epochs=max_epoch, batch_size=batch_size)
dcgan = DCGAN(batch_size=batch_size, channel=channel_num, input_pipeline=data_pipeline)

# init TF
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    writer = tf.summary.FileWriter('./graphs', sess.graph)

    epoch = 0
    batch_count = data_count // batch_size

    for epoch in range(max_epoch):

        for i in range(batch_count):

            #z = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            z = np.random.normal(loc=0.0, scale=0.5, size=[batch_size, 100])
            zero_label = np.zeros([batch_size, 1], dtype=np.float32)
            one_label = np.ones([batch_size, 1], dtype=np.float32)
            zero_label +=label_smoothing
            one_label -= label_smoothing

            _, d_loss_fake, ds = sess.run([dcgan.d_optim, dcgan.loss, dcgan.d_sum],
                feed_dict={dcgan.is_real: False, dcgan.input_noise:z, dcgan.input_label:zero_label})

            _, d_loss_real, ds = sess.run([dcgan.d_optim, dcgan.loss, dcgan.d_sum],
                feed_dict={dcgan.is_real: True, dcgan.input_noise:z, dcgan.input_label:one_label})

            _, g_loss1, gen, gs = sess.run([dcgan.g_optim, dcgan.loss, dcgan.generated, dcgan.g_sum],
                feed_dict={dcgan.is_real: False, dcgan.input_noise:z, dcgan.input_label:one_label})

            _, g_loss2, gen, gs = sess.run([dcgan.g_optim, dcgan.loss, dcgan.generated, dcgan.g_sum],
                feed_dict={dcgan.is_real: False, dcgan.input_noise:z, dcgan.input_label:one_label})

            d_loss = d_loss_fake + d_loss_real
            g_loss = g_loss1 + g_loss2

            writer.add_summary(ds, epoch * batch_count + i)
            writer.add_summary(gs, epoch * batch_count + i)

            print('EPOCH {}'.format(epoch), '[{}/{}] D_Loss: {}, G_Loss: {}'.format(i, batch_count, d_loss, g_loss))

            if i % save_image_interval == 0:
                # de-preprocess image & save
                z = np.random.normal(loc=0.0, scale=0.1, size=[batch_size, 100])
                gen = sess.run(dcgan.generated, feed_dict={dcgan.input_noise:z})
                if channel_num > 1:
                    gen = np.reshape(gen, [batch_size, 64, 64, channel_num])
                else:
                    gen = np.reshape(gen, [batch_size, 64, 64])
                gen_img = gen + 1.0 * 127.0
                if not os.path.exists('gen_celeb'):
                    os.makedirs('gen_celeb')
                image_util.save_images('gen_celeb/{}_{}.png'.format(epoch, i), gen_img[0:64], [8,8])
