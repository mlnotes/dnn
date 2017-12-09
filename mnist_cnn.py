"""CNN for mnist."""

import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = '/tmp/mnist_cnn'
STEPS = 10000
pkeep = tf.placeholder(tf.float32, name='pkeep')

def build_conv_layer(in_tensor, filter_height, filter_width, out_channels, stride, name):
  in_channels = in_tensor.get_shape().dims[3].value
  with tf.name_scope(name):
    filter = tf.Variable(tf.truncated_normal([filter_height,
                                              filter_width,
                                              in_channels,
                                              out_channels],
                                             stddev=0.1), name='filter')
    bias = tf.Variable(tf.ones([out_channels])/10, name='bias')
    conv = tf.nn.conv2d(in_tensor, filter, strides=[1, stride, stride, 1], padding='SAME', name='conv')
    return tf.nn.relu(conv + bias)

def build_fc_layer(in_tensor, out_size, name):
  in_size = in_tensor.get_shape().dims[1].value

  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1), name='weights')
    b = tf.Variable(tf.zeros([out_size]), name='bias')
    #output = tf.nn.softmax(tf.matmul(in_tensor, w) + b)
    output = tf.nn.relu(tf.matmul(in_tensor, w) + b)
    return tf.nn.dropout(output, pkeep)

def build_softmax_layer(in_tensor, labels, name):
  in_size = in_tensor.get_shape().dims[1].value
  out_size = labels.get_shape().dims[1].value

  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1), name='weights')
    b = tf.Variable(tf.zeros([out_size]), name='bias')
    output = tf.nn.softmax(tf.matmul(in_tensor, w) + b)
    return output


def build_accuracy(output, labels):
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      return accuracy

def build_loss(output, labels):
  with tf.name_scope('loss'):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=output, labels=labels))


def build_train_step(loss):
  with tf.name_scope('train'):
    lr = tf.placeholder(tf.float32, name='learning_rate')
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    return lr, train_step


def build_cnn_network(features, labels):
  l1_out = build_conv_layer(features, 5, 5, 4, 1, 'l1_conv')
  l2_out = build_conv_layer(l1_out, 5, 5, 8, 2, 'l2_conv')
  l3_out = build_conv_layer(l2_out, 4, 4, 12, 2, 'l3_conv')
  # reshap
  conv_out = tf.reshape(l3_out, shape=[-1, 7 * 7 * 12])
  fc_out = build_fc_layer(conv_out, 200, 'fc_layer')
  output = build_softmax_layer(fc_out, labels, 'softmax_layer')

  accuracy = build_accuracy(output, labels)
  loss = build_loss(output, labels)
  lr, train_step = build_train_step(loss)

  return accuracy, loss, lr, train_step, l1_out, l2_out, l3_out


def main():
  features = tf.placeholder(tf.float32, [None, 28, 28, 1])
  labels = tf.placeholder(tf.float32, [None, 10])
  accuracy, loss, lr, train_step, l1_out, l2_out, l3_out = build_cnn_network(features, labels)

  tf.summary.scalar('accuracy', accuracy)
  tf.summary.scalar('loss', loss)
  tf.summary.scalar('lr', lr)
  tf.summary.image('l1_out', l1_out)

  # l2_out [None, 14, 14, 8]
  tf.summary.image('l2_out', tf.reshape(l2_out, [-1, 14, 28, 4]))

  # l3_out [None, 7, 7, 12]
  tf.summary.image('l3_out', tf.reshape(l3_out, [-1, 7, 21, 4]))
  merged = tf.summary.merge_all()

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(LOG_DIR + '/test')

  max_lr = 0.003
  min_lr = 0.0001
  decay_speed = 2000.0

  mnist = input_data.read_data_sets('MNIST_data/', one_hot=True, reshape=False)
  # Train, reading data
  for i in range(STEPS):
    images, digits = mnist.train.next_batch(100)
    curr_lr = min_lr + (max_lr - min_lr) * math.exp(-i / decay_speed)
    sess.run(train_step, feed_dict={features: images,
                                    labels: digits,
                                    lr: curr_lr,
                                    pkeep: 0.75})

    if i % 10 == 0:
      summary = sess.run(merged, feed_dict={features: mnist.test.images,
                                            labels: mnist.test.labels,
                                            lr: curr_lr,
                                            pkeep: 1.0})
      test_writer.add_summary(summary, i)

    summary = sess.run(merged, feed_dict={features: images,
                                          labels: digits,
                                          lr: curr_lr,
                                          pkeep: 1.0})
    train_writer.add_summary(summary, i)



if __name__ == '__main__':
  main()
