"""Customized estimator."""

import functools
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner
from tensorflow.examples.tutorials.mnist import input_data


def build_network(features, is_training):
 # with tf.variable_scope('network'):
  conv1 = tf.layers.conv2d(inputs=features, filters=2, kernel_size=5,
                           strides=1, padding='same', name='conv1', trainable=True)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2,
                                  strides=2, padding='same', name='max_pool1')
  conv2 = tf.layers.conv2d(inputs=pool1, filters=4, kernel_size=4,
                           strides=1, padding='same', name='conv2')
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2,
                                  strides=2, padding='same', name='max_pool2')

  fc_input = tf.reshape(pool2, [-1, 7 * 7 * 4])

  fc1 = tf.layers.dense(inputs=fc_input, units=200, activation=tf.nn.relu, name='fc1')
  dropout = tf.layers.dropout(inputs=fc1, rate=0.75, training=is_training, name='dropout')
  fc2 = tf.layers.dense(inputs=dropout, units=10, name='fc2')
  return fc2


def make_input_fn(images, digits, num_epochs=None):
  return tf.estimator.inputs.numpy_input_fn(
      x={'x': images},
      y=digits.astype(int),
      num_epochs=num_epochs,
      shuffle=True)

def make_train_op(loss, hparams):
  lr_decay_fn = functools.partial(
      tf.train.exponential_decay,
      decay_rate=0.5,
      decay_steps=1000,
      staircase=False)

  return tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      optimizer=tf.train.AdamOptimizer,
      learning_rate=hparams.learning_rate,
      learning_rate_decay_fn=lr_decay_fn)

def make_eval_metric_ops(labels, predictions):
  return {
      'Accuracy': tf.metrics.accuracy(
          labels=labels,
          predictions=predictions,
          name='accuracy')
  }

def model_fn(features, labels, mode, params):
  is_training = mode == ModeKeys.TRAIN
  logits = build_network(features['x'], is_training)
  predictions = tf.argmax(logits, 1)
  loss = tf.losses.sparse_softmax_cross_entropy(
      labels=labels,
      logits=logits)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=make_train_op(loss, params),
      eval_metric_ops=make_eval_metric_ops(labels, predictions))

def make_estimator(run_config, hparams):
  return tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params=hparams)


def experiment_fn(run_config, hparams):
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False, reshape=False)
  return tf.contrib.learn.Experiment(
      estimator=make_estimator(run_config, hparams),
      train_input_fn=make_input_fn(mnist.train.images, mnist.train.labels),
      eval_input_fn=make_input_fn(mnist.validation.images, mnist.validation.labels),
      train_steps=5000)


def main():
  run_config = tf.contrib.learn.RunConfig(
      model_dir='/tmp/cus_model',
      save_summary_steps=1,
      save_checkpoints_steps=10)

  hparams = tf.contrib.training.HParams(
      learning_rate=0.002)

  learn_runner.run(experiment_fn=experiment_fn,
                   run_config=run_config,
                   hparams=hparams,
                   schedule='train_and_evaluate')


if __name__ == '__main__':
  main()
