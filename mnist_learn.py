"""CNN to recognize hand write digits."""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = '/tmp/mnist_logs'
LR = 0.005

def build_layer(in_tensor, out_size, name):
  in_size = in_tensor.get_shape().dims[1].value
  with tf.name_scope(name):
    w = tf.truncated_normal([in_size, out_size], stddev=0.1, name='weights')
    # w = tf.Variable(tf.zeros([in_size, out_size]), name='weights')
    b = tf.Variable(tf.zeros([out_size]), name='bias')
    # return tf.nn.relu(tf.matmul(in_tensor, w) + b)
    return tf.nn.sigmoid(tf.matmul(in_tensor, w) + b)

def build_output_layer(in_tensor, out_size, name):
  in_size = in_tensor.get_shape().dims[1].value
  with tf.name_scope(name):
    #w = tf.truncated_normal([in_size, out_size], stddev=0.1, name='weights')
    w = tf.Variable(tf.zeros([in_size, out_size]), name='weights')
    b = tf.Variable(tf.zeros([out_size]), name='bias')
    return tf.nn.softmax(tf.matmul(in_tensor, w) + b)

def build_loss(out_tensor, label):
  with tf.name_scope('loss'):
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(out_tensor * label, reduction_indices=[1]))
    return cross_entropy

def build_accuracy(output, labels):
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_predition'):
      correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      return accuracy

def build_train_step(loss):
  with tf.name_scope('train'):
    # train_step = tf.train.GradientDescentOptimizer(LR).minimize(loss)
    train_step = tf.train.AdamOptimizer(LR).minimize(loss)
  return train_step

def build_network(features, labels, hidden_layers=[]):
  label_dim = labels.get_shape().dims[1].value
  out = features
  for i in range(len(hidden_layers)):
    out = build_layer(out, hidden_layers[i], 'l%d_layer' % (i+1))
  output = build_output_layer(out, label_dim, 'l%d_layer' % (len(hidden_layers) + 1))
  accuracy = build_accuracy(output, labels)
  loss = build_loss(output, labels)
  train_step = build_train_step(loss)
  return accuracy, loss, train_step

def build_cnn_network(features, labels):
  pass

def train(features, labels, train_step, merged_summary, steps=1000):
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(LOG_DIR + '/test')

  mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
  for i in range(steps):
    images, digits = mnist.train.next_batch(100)
    if i % 10 == 0:
      summary = sess.run(merged_summary, feed_dict={features: mnist.test.images,
                                                    labels: mnist.test.labels})
      test_writer.add_summary(summary, i)
    summary, _ = sess.run([merged_summary, train_step],
                          feed_dict={features: images, labels: digits})
    train_writer.add_summary(summary, i)


def main():
  features = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='features')
  labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
  # accuracy, loss, train_step = build_network(features, labels, [200, 100, 60, 30])
  accuracy, loss, train_step = build_network(features, labels, [])

  tf.summary.scalar('accuracy', accuracy)
  tf.summary.scalar('loss', loss)
  merged = tf.summary.merge_all()

  train(features, labels, train_step, merged, 10000)


if __name__ == '__main__':
  main()
