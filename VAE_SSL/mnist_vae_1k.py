"""
    training mnist with first 1000 labeled data
    & Auto-encoder with un-labeled data

"""

import numpy as np
import tensorflow as tf
from mnist_data import load_mnist, load_mnist_t10k

def dense(x, input_num, output_num, name="dense", with_var=False):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([input_num, output_num], stddev=0.02))
        b = tf.Variable(tf.zeros([output_num]))
        if with_var:
            return tf.matmul(x, w) + b, w, b
        return tf.matmul(x, w) + b

def batch_norm(input_v, train=True, name="bn"):
    return tf.contrib.layers.batch_norm(input_v, decay=0.9, updates_collections=None,
                    epsilon=1e-5, scale=True, is_training=train, scope=name)

#
#   Auto-encoder
#

ae_batch_size = 200
ae_input_size = 28 * 28
ae_hidden1_size = 300
ae_hidden2_size = 200
ae_hidden3_size = 100
ae_neck_size = 50
ae_train_size = 60000
ae_max_epoch = 300

ae_x = tf.placeholder(shape=[ae_batch_size, ae_input_size], dtype=tf.float32, name="ae_input")

ae_hidden1 = dense(ae_x, ae_input_size, ae_hidden1_size, "ae_e1")
ae_hidden1 = tf.nn.relu(ae_hidden1, "e_relu1")

ae_hidden2 = dense(ae_hidden1, ae_hidden1_size, ae_hidden2_size, "ae_e2")
ae_hidden2 = tf.nn.relu(ae_hidden2, "e_relu2")

ae_hidden3 = dense(ae_hidden2, ae_hidden2_size, ae_hidden3_size, "ae_e3")
ae_hidden3 = tf.nn.relu(ae_hidden3, "e_relu3")

ae_z_mean = dense(ae_hidden3, ae_hidden3_size, ae_neck_size, "ae_mean")
ae_z_stddev = dense(ae_hidden3, ae_hidden3_size, ae_neck_size, "ae_stddev")

ae_latent_feature = ae_z_mean + ae_z_stddev

#ae_noise = tf.random_normal([ae_batch_size, ae_neck_size], mean=0.0, stddev=1.0, dtype=tf.float32)
#ae_latent_noisy_feature = ae_z_mean + (ae_z_stddev * ae_noise)
ae_latent_noisy_feature = ae_z_mean + ae_z_stddev

ae_latent_loss = 0.5 * tf.reduce_sum(tf.square(ae_z_mean) + tf.square(ae_z_stddev) - tf.log(tf.square(ae_z_stddev)) - 1, 1, name="latent_loss")

ae_hidden4 = dense(ae_latent_noisy_feature, ae_neck_size, ae_hidden3_size, "ae_d1")
ae_hidden4 = batch_norm(ae_hidden4, name="e_norm4")
ae_hidden4 = tf.nn.relu(ae_hidden4, "d_relu4")

ae_hidden5 = dense(ae_hidden4, ae_hidden3_size, ae_hidden2_size, "ae_d2")
ae_hidden5 = batch_norm(ae_hidden5, name="e_norm5")
ae_hidden5 = tf.nn.relu(ae_hidden5, "d_relu5")

ae_hidden6 = dense(ae_hidden5, ae_hidden2_size, ae_hidden1_size, "ae_d3")
ae_hidden6 = batch_norm(ae_hidden6, name="e_norm6")
ae_hidden6 = tf.nn.relu(ae_hidden6, "d_relu6")

ae_output = dense(ae_hidden6, ae_hidden1_size, ae_input_size, "ae_d4")
ae_output = tf.nn.tanh(ae_output, "tanh7")

ae_reconstruction_loss = tf.reduce_sum(tf.square(ae_output - ae_x), 1, name="reconstruction_loss")
#ae_reconstruction_loss = tf.reduce_sum(tf.losses.log_loss(ae_x, ae_output, reduction=None), 1, name="reconstruction_loss")


ae_loss = tf.reduce_mean(ae_latent_loss + ae_reconstruction_loss)
#ae_loss = tf.reduce_mean(ae_reconstruction_loss)

ae_trainer = tf.train.AdamOptimizer(learning_rate = 0.001)
ae_optimize = ae_trainer.minimize(ae_loss)


#
#   Classifier
#

batch_size = 200
input_size = 50
hidden1_size = 300
hidden2_size = 150
hidden3_size = 70
output_size = 10
train_size = 1000
test_size = 10000
max_epoch = 300

label = tf.placeholder(shape=[batch_size], dtype=tf.int64, name="label")
is_training = tf.placeholder(dtype=tf.bool, name="is_training")

hidden1, weights1, biases1 = dense(ae_latent_feature, input_size, hidden1_size, "fc1", with_var=True)
hidden1 = batch_norm(hidden1, train=is_training, name="fc_norm1")
hidden1 = tf.nn.relu(hidden1, "fc_relu1")
#hidden1 = tf.layers.dropout(hidden1, rate=0.5, training=is_training, name="dropout1")

hidden2, weights2, biases2 = dense(hidden1, hidden1_size, hidden2_size, "fc2", with_var=True)
hidden2 = batch_norm(hidden2, train=is_training, name="fc_norm2")
hidden2 = tf.nn.relu(hidden2, "fc_relu2")
#hidden2 = tf.layers.dropout(hidden2, rate=0.5, training=is_training, name="dropout2")

hidden3, weights3, biases3 = dense(hidden2, hidden2_size, hidden3_size, "fc3", with_var=True)
hidden3 = batch_norm(hidden3, train=is_training, name="fc_norm3")
hidden3 = tf.nn.relu(hidden3, "fc_relu3")
#hidden3 = tf.layers.dropout(hidden3, rate=0.5, training=is_training, name="dropout3")

output, weights4, biases4 = dense(hidden3, hidden3_size, output_size, "fc4", with_var=True)
pred = tf.nn.softmax(output)

onehot = tf.one_hot(label, output_size, axis=1)
with tf.name_scope("cross_entropy_loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=onehot))

trainer = tf.train.AdamOptimizer(learning_rate=0.001)
optimize = trainer.minimize(loss, var_list=[weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4])

correct = tf.equal(tf.argmax(pred, axis=1), label)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

images, labels = load_mnist('./mnist')
t_images, t_labels = load_mnist_t10k('./mnist')
images = images / 127.0 - 1.0
t_images = t_images / 127.0 - 1.0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    init = tf.global_variables_initializer()
    session.run(init)

    writer = tf.summary.FileWriter('./graphs', session.graph)

    # training AE first
    ae_batch_count = ae_train_size // ae_batch_size
    for ae_epoch in range(ae_max_epoch):
        ae_total_loss = 0
        for i in range(ae_batch_count):
            ae_index = i * ae_batch_size
            img = np.reshape(images[ae_index:ae_index+ae_batch_size], [ae_batch_size, 28 * 28])
            _, loss_v = session.run([ae_optimize, ae_loss], feed_dict= {ae_x: img})
            ae_total_loss += loss_v
        print('AE Epoch {}'.format(ae_epoch), 'loss:', ae_total_loss / ae_batch_count)


    # training classifier
    batch_count = train_size // batch_size
    for epoch in range(max_epoch):
        total_loss = 0
        for i in range(batch_count):
            img = np.reshape(images[i*batch_size:(i+1)*batch_size], [batch_size, 28 * 28])
            lbl = (labels[i*batch_size:(i+1)*batch_size])
            _, loss_v = session.run([optimize, loss], feed_dict= {ae_x: img, label: lbl, is_training: True})
            total_loss += loss_v
        print('Epoch {}'.format(epoch), 'loss:', total_loss / batch_count)

        # accuracy in training set
        total_acc = 0
        test_count = train_size // batch_size
        for a in range(test_count):
            index = a * batch_size
            img = np.reshape(images[index:index+batch_size], [batch_size, 28 * 28])
            lbl = labels[index:index+batch_size]
            acc = session.run(accuracy, feed_dict={ae_x:img, label:lbl, is_training: False})
            total_acc += acc
        total_acc = total_acc / test_count
        print('  Training Score: ', total_acc * 100, '%')

        # accuracy in test set
        total_acc = 0
        test_count = test_size // batch_size
        for a in range(test_count):
            index = a * batch_size
            img = np.reshape(t_images[index:index+batch_size], [batch_size, 28 * 28])
            lbl = t_labels[index:index+batch_size]
            acc = session.run(accuracy, feed_dict={ae_x:img, label:lbl, is_training: False})
            total_acc += acc
        total_acc = total_acc / test_count
        print('  Test Score: ', total_acc * 100, '%')
