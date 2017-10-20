"""
    DCGAN semi-supervised learning implementation by hkroh
    September 28, 2017

"""

import numpy as np
import tensorflow as tf
import mnist_data
import image_util
import os

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

def fully_connected(x, output_size, name="fc", stddev=0.02, with_var=False):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name or "fully_connected"):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(0.0))
        if with_var:
            return tf.matmul(x, w) + b, w, b
        else:
            return tf.matmul(x, w) + b

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
        self.max_filter = 128
        self.input_channel = channel

        # placeholder
        self.input_noise = tf.placeholder(shape=[self.batch_size, 100], dtype=tf.float32, name="input")

        with tf.variable_scope("Generator"):
            net = fully_connected(self.input_noise, 7 * 7 * self.max_filter, "linear0")
            net = tf.reshape(net, [self.batch_size, 7, 7, self.max_filter], name="reshape")
            net = batch_norm(net, name="batch_norm0")
            net = tf.nn.relu(net, name="relu0")

            net = deconv2d(net, [self.batch_size, 14, 14, self.max_filter//2],  name="deconv1")
            net = batch_norm(net, name="batch_norm1")
            net = tf.nn.relu(net, name="relu1")

            net = deconv2d(net, [self.batch_size, 28, 28, self.input_channel], name="deconv2")
            self.generated = tf.nn.tanh(net)

        with tf.variable_scope("Discriminator"):
            self.inputD = self.generated
            self.labelD = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.float32, name="labelD")
            self.labelClass = tf.placeholder(shape=[self.batch_size], dtype=tf.int64, name="labelClass")
            self.is_labeled = tf.placeholder(tf.bool, name="is_labeled")

            net = conv2d(self.inputD, self.max_filter//2, name="conv5")
            net = batch_norm(net, name="batch_norm5")
            net = lrelu(net, name="lrelu5")

            net = conv2d(net, self.max_filter, name="conv6")
            net = batch_norm(net, name="batch_norm6")
            net = lrelu(net, name="lrelu6")

            self.flatten = tf.reshape(net, [self.batch_size, -1])
            self.output = fully_connected(self.flatten, 1, "output")
            self.pred_real = tf.nn.sigmoid(self.output, "pred_real")

            self.output_class = fully_connected(self.flatten, 10, "output_class")
            self.pred_class = tf.nn.softmax(self.output_class, name="pred_class")

            self.label_onehot = tf.one_hot(self.labelClass, 10, axis=1)

            self.gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.labelD))
            self.cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_class, labels=self.label_onehot))

        self.loss = tf.cond(self.is_labeled, lambda: self.gan_loss + self.cls_loss, lambda: self.gan_loss)

        self.correct = tf.equal(tf.argmax(self.pred_class, axis=1), self.labelClass)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        self.d_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss, var_list=self.d_vars)
        self.g_vars = [var for var in t_vars if 'Generator' in var.name]
        self.g_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss, var_list=self.g_vars)

        self.loss_sum = tf.summary.scalar('loss', self.loss)
        self.sum = tf.summary.merge([self.loss_sum])

#
#   training hyper-parameter
#
batch_size = 100
max_epoch = 500
dcgan = DCGAN(batch_size=batch_size, channel=1)

# load MNIST data
images, labels = mnist_data.load_mnist('./mnist')
input_img = []
for i in range(60000):
    input_img.append(images[i])
input_img = np.array(input_img)

# preprocess images
input_img = input_img / 127.0 - 1.0
input_img = np.reshape(input_img, [60000, 28, 28, 1])

t_images, t_labels = mnist_data.load_mnist_t10k('./mnist')
t_images = t_images / 127.0 - 1.0
t_images = t_images.reshape([10000, 28, 28, 1])

labeled_image = input_img[:1000]
labeled_label = labels[:1000]

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
    for epoch in range(max_epoch):

        for i in range(batch_count):

            z = np.random.normal(0.0, 1.0, size=[batch_size, 100])

            batch_input = input_img[i*batch_size:(i+1)*batch_size]

            label_real = np.ones([batch_size, 1], np.float32)
            label_real -= 0.1
            label_fake = np.zeros([batch_size, 1], np.float32)
            label_fake += 0.1

            dummy_label = np.ones([batch_size], np.int64)

            # training D
            _, d_loss1, ds = sess.run([dcgan.d_optim, dcgan.loss, dcgan.sum],
                feed_dict={dcgan.input_noise:z, dcgan.labelD: label_fake, dcgan.is_labeled: False, dcgan.labelClass:dummy_label})

            _, d_loss2, ds = sess.run([dcgan.d_optim, dcgan.loss, dcgan.sum],
                feed_dict={dcgan.inputD:batch_input, dcgan.labelD: label_real, dcgan.is_labeled: False, dcgan.labelClass:dummy_label})

            # training D with labeled data
            step = epoch * batch_count + i
            labeled_start_index = step % 10
            labeled_train_image = labeled_image[labeled_start_index*100:(labeled_start_index+1)*100]
            labeled_train_label = labeled_label[labeled_start_index*100:(labeled_start_index+1)*100]
            _, d_loss3, ds = sess.run([dcgan.d_optim, dcgan.loss, dcgan.sum],
                feed_dict={dcgan.inputD:labeled_train_image, dcgan.labelD:label_real, dcgan.is_labeled:True, dcgan.labelClass:labeled_train_label})

            # training G
            _, g_loss1, gen, gs = sess.run([dcgan.g_optim, dcgan.loss, dcgan.generated, dcgan.sum],
                feed_dict={dcgan.input_noise:z, dcgan.labelD: label_real, dcgan.is_labeled: False, dcgan.labelClass:dummy_label})

            _, g_loss2, gen, gs = sess.run([dcgan.g_optim, dcgan.loss, dcgan.generated, dcgan.sum],
                feed_dict={dcgan.input_noise:z, dcgan.labelD: label_real, dcgan.is_labeled: False, dcgan.labelClass:dummy_label})

            d_loss = d_loss1 + d_loss2 + d_loss3
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

        # testing
        batch_count_test = 10000 // batch_size
        total_acc = 0.0
        for i in range(batch_count_test):
            batch_index = i * batch_size
            batch_input = t_images[batch_index:batch_index+batch_size]
            batch_label = t_labels[batch_index:batch_index+batch_size]
            batch_input = batch_input.reshape([batch_size, 28, 28, 1])
            acc = sess.run(dcgan.accuracy, feed_dict={dcgan.inputD:batch_input, dcgan.labelClass:batch_label, dcgan.is_labeled:True})
            total_acc += acc
        print('  Test acc:', total_acc / batch_count_test * 100, '%')
