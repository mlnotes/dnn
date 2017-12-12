"""MNIST classification with tensorflow estimator."""

import math
import tensorflow as tf
from tensorflow.contrib.learn import learn_runner
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = '/tmp/est_logs'

def build_cnn_network(features):
  conv1 = tf.layers.conv2d(inputs=features, filters=2, kernel_size=5,
                           strides=1, padding='same', name='conv1')
  max_pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2,
                                      strides=2, padding='same', name='max_pool1')
  conv2 = tf.layers.conv2d(inputs=max_pool1, filters=4, kernel_size=4,
                           strides=1, padding='same', name='conv2')
  max_pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2,
                                      strides=2, padding='same', name='max_pool2')
  fc_input = tf.reshape(max_pool2, [-1, 7 * 7 * 4])
  fc1 = tf.layers.dense(inputs=fc_input, units=256, activation=tf.nn.relu, name='fc1')

  training = tf.placeholder(tf.bool, name='is_training')
  dropout = tf.layers.dropout(inputs=fc1, rate=0.75, training=training, name='dropout')

  fc2 = tf.layers.dense(inputs=dropout, units=10, name='fc2')
  return training, fc2

def build_loss(labels, logits):
  with tf.name_scope('loss'):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits))

def build_accuracy(labels, logits):
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    with tf.name_scope('accuracy'):
      return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def build_train_step(loss, lr):
  with tf.name_scope('train'):
    return tf.train.AdamOptimizer(lr).minimize(loss)

def train(train_step, features, labels, lr, training, merged_summary):
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False)
  max_lr = 0.003
  min_lr = 0.0001
  lr_decay_speed = 2000.0

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)
    for i in range(10000):
      images, digits = mnist.train.next_batch(100)
      curr_lr = min_lr + (max_lr - min_lr) * math.exp(-i / lr_decay_speed)
      sess.run(train_step, feed_dict={features: images,
                                      labels: digits,
                                      lr: curr_lr,
                                      training: True})
      summary = sess.run(merged_summary, feed_dict={features: images,
                                                    labels: digits,
                                                    lr: curr_lr,
                                                    training: False})
      train_writer.add_summary(summary, i)
      if i % 10 == 0:
        summary = sess.run(merged_summary, feed_dict={features: mnist.test.images,
                                                      labels: mnist.test.labels,
                                                      lr: curr_lr,
                                                      training: False})
        test_writer.add_summary(summary, i)


# estimator
def make_input_fn(images, digits, num_epochs=None):
  return tf.estimator.inputs.numpy_input_fn(
      x={'x': images},
      y=digits.astype(int),
      num_epochs=num_epochs,
      shuffle=True)

def make_estimator():
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False, reshape=True)
  feature_columns = [
      tf.feature_column.numeric_column('x', shape=[28 * 28 * 1])
  ]
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[200, 100, 60, 30],
                                          n_classes=10,
                                          model_dir='/tmp/est_model')

  classifier.train(input_fn=make_input_fn(mnist.train.images, mnist.train.labels),
                   steps=5000)
  accuracy_score = classifier.evaluate(input_fn=make_input_fn(mnist.test.images,
                                                              mnist.test.labels,
                                                              num_epochs=1))

  print 'HF: acc:', accuracy_score


def make_experiment(run_config, params):
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False, reshape=True)
  feature_columns = [
      tf.feature_column.numeric_column('x', shape=[28 * 28 * 1])
  ]
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[200, 100, 60, 30],
                                          n_classes=10,
                                          config=run_config)

  return tf.contrib.learn.Experiment(
      estimator=classifier,
      train_input_fn=make_input_fn(mnist.train.images, mnist.train.labels),
      eval_input_fn=make_input_fn(mnist.validation.images, mnist.validation.labels, num_epochs=1),
      train_steps=5000)


def main():
  run_config = tf.contrib.learn.RunConfig()
  run_config = run_config.replace(
      model_dir='/tmp/est_model',
      save_summary_steps=1,
      save_checkpoints_steps=10
  )

  learn_runner.run(experiment_fn=make_experiment,
                   run_config=run_config,
                   schedule='train_and_evaluate')


def main1():
  features = tf.placeholder(tf.float32, [None, 28, 28, 1], name='features')
  labels = tf.placeholder(tf.float32, [None, 10], name='labels')
  lr = tf.placeholder(tf.float32, name='learning_rate')

  training, logits = build_cnn_network(features)
  accuracy = build_accuracy(labels, logits)
  loss = build_loss(labels, logits)
  train_step = build_train_step(loss, lr)

  tf.summary.scalar('accuracy', accuracy)
  tf.summary.scalar('loss', loss)
  tf.summary.scalar('lr', lr)

  train(train_step, features, labels, lr, training, tf.summary.merge_all())

if __name__ == '__main__':
  main()
