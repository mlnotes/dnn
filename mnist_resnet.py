"""ResNet for MNIST data."""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = '/tmp/resnet'
STEPS = 10000
total_layers = 25
units_between_stride = total_layers / 5

def build_res_net_unit(inputs, i):
  with tf.variable_scope("res_unit_%d" % i):
    part1 = tf.contrib.slim.batch_norm(inputs)
    part2 = tf.nn.relu(part1)
    part3 = tf.contrib.slim.conv2d(part2, 64, [3, 3], activation_fn=None)
    part4 = tf.contrib.slim.batch_norm(part3)
    part5 = tf.nn.relu(part4)
    part6 = tf.contrib.slim.conv2d(part5, 64, [3, 3], activation_fn=None)
    outputs = inputs + part6
    return outputs


def build_res_net(features):
  layer = tf.contrib.slim.conv2d(features, 64, [3, 3],
                                 normalizer_fn=tf.contrib.slim.batch_norm,
                                 scope='conv_%d' % 0)
  for i in range(5):
    for j in range(units_between_stride):
      layer = build_res_net_unit(layer, j + (i * units_between_stride))
    layer = tf.contrib.slim.conv2d(layer, 64, [2, 2],
                                   normalizer_fn=tf.contrib.slim.batch_norm,
                                   scope='conv_s_%d' % i)
  top = tf.contrib.slim.conv2d(layer, 10, [3, 3], stride=[2, 2],
                               normalizer_fn=tf.contrib.slim.batch_norm,
                               activation_fn=None,
                               scope='conv_top')

  pool = tf.layers.average_pooling2d(inputs=top, pool_size=2, strides=1,
                                     padding='SAME', name='avg_pool')

  fc_input = tf.reshape(pool, [-1, 14 * 14 * 10])
  fc1 = tf.layers.dense(inputs=fc_input, units=256, activation=tf.nn.relu, name='fc1')
  fc2 = tf.layers.dense(inputs=fc1, units=10, name='fc2')
  return fc2


def build_accuracy(labels, logits):
  with tf.variable_scope('accuracy'):
    with tf.variable_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    with tf.variable_scope('accuracy'):
      return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def build_loss(labels, logits):
  with tf.variable_scope('loss'):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits))


def build_train_step(loss):
  with tf.variable_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    return train_step


def main():
  features = tf.placeholder(tf.float32, [None, 28, 28, 1], name='features')
  labels = tf.placeholder(tf.float32, [None, 10], name='labels')

  logits = build_res_net(features)
  accuracy = build_accuracy(labels, logits)
  loss = build_loss(labels, logits)
  train_step = build_train_step(loss)

  sess = tf.Session()
  with sess.as_default():
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test')

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', loss)
    merged_summary = tf.summary.merge_all()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False)
    for i in range(STEPS):
      images, digits = mnist.train.next_batch(128)
      sess.run(train_step, feed_dict={features: images, labels: digits})

      if i % 100 == 0:
        test_sumary = sess.run(merged_summary, feed_dict={features: mnist.validation.images,
                                                          labels: mnist.validation.labels})
        test_writer.add_summary(test_summary, i)
        train_sumary = sess.run(merged_summary, feed_dict={features: images,
                                                          labels: digits})
        train_writer.add_summary(train_summary, i)




if __name__ == '__main__':
  main()
