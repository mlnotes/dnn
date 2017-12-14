"""Customized estimator."""

import functools
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner
from tensorflow.examples.tutorials.mnist import input_data

total_layers = 25
units_between_stride = total_layers / 5

def build_res_net_block(inputs, i):
  with tf.variable_scope("res_unit_%d" % i):
    bn = tf.contrib.slim.batch_norm(inputs)
    relu = tf.nn.relu(bn)
    conv = tf.contrib.slim.conv2d(relu, 64, [3, 3], activation_fn=None)
    bn = tf.contrib.slim.batch_norm(conv)
    relu = tf.nn.relu(bn)
    conv = tf.contrib.slim.conv2d(relu, 64, [3, 3], activation_fn=None)
    outputs = inputs + conv
    return outputs


def build_res_net(features):
  layer = tf.contrib.slim.conv2d(features, 64, [3, 3],
                                 normalizer_fn=tf.contrib.slim.batch_norm,
                                 scope='conv_%d' % 0)
  for i in range(5):
    for j in range(units_between_stride):
      layer = build_res_net_block(layer, j + (i * units_between_stride))
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
  fc = tf.layers.dense(inputs=fc_input, units=10, name='fc')
  return fc

def build_network(features, is_training):
  return build_res_net(features)


def make_input_fn(images, digits, num_epochs=None):
  return tf.estimator.inputs.numpy_input_fn(
      x={'x': images},
      y=digits.astype(int),
      num_epochs=num_epochs,
      shuffle=True)

def make_train_op(loss, hparams):
  lr_decay_fn = functools.partial(
      tf.train.exponential_decay,
      decay_rate=0.7,
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
      train_steps=10001)


def main():
  run_config = tf.contrib.learn.RunConfig(
      model_dir='/tmp/res_model1',
      save_summary_steps=10,
      save_checkpoints_steps=10)

  hparams = tf.contrib.training.HParams(
      learning_rate=0.003)

  learn_runner.run(experiment_fn=experiment_fn,
                   run_config=run_config,
                   hparams=hparams,
                   schedule='train_and_evaluate')


if __name__ == '__main__':
  main()
