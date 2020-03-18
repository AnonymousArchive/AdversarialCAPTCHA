from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

from cleverhans.compat import flags
from cleverhans.loss import CrossEntropy
from cleverhans.dataset import DISK
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN



FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64

def zip_tests(x1, y1, x2, y2, ratio=0.5):
    x = np.zeros((10000, 28, 28, 1), dtype='float64')
    y = np.zeros((10000, 10), dtype='float64')
    ratio = int(ratio * 100)
    c = []
    for i in range(10000):
        if i % 100 <= ratio:
            c.append((x1[i], y1[i]))
        else:
            c.append((x2[i], y2[i]))
    random.shuffle(c)
    for i in range(10000):
        x[i] = c[i][0]
        y[i] = c[i][1]
    return x, y

def zip_trains(x1, y1, x2, y2, ratio=0.5):
    x = np.zeros((60000, 28, 28, 1), dtype='float32')
    y = np.zeros((60000, 10), dtype='float32')
    ratio = int(ratio * 100)
    c = []
    for i in range(60000):
        if i % 100 <= ratio:
            c.append((np.asarray(x1[i]).reshape((28, 28, 1)), np.asarray(y1[i]).reshape((10))))
        else:
            c.append((np.asarray(x2[i]).reshape((28, 28, 1)), np.asarray(y2[i]).reshape((10))))
    random.shuffle(c)
    for i in range(60000):
        x[i] = c[i][0]
        y[i] = c[i][1]
    return x, y

def get_train(s):
    path = 'images/' + s
    mnist = DISK(path, train_start=0, train_end=60000, test_start=0,
                     test_end=10000)
    x_train, y_train = mnist.get_set('train')
    return x_train, y_train

def get_test(s):
    path = 'images/' + s
    mnist = DISK(path, train_start=0, train_end=60000, test_start=0,
                     test_end=10000)
    x_test, y_test = mnist.get_set('test')
    return x_test, y_test


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE,
                   clean_train=CLEAN_TRAIN,
                   testing=False,
                   backprop_through_attack=BACKPROP_THROUGH_ATTACK,
                   nb_filters=NB_FILTERS, num_threads=None,
                   label_smoothing=0.1):
  """
  MNIST cleverhans tutorial
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param clean_train: perform normal training on clean examples only
                      before performing adversarial training.
  :param testing: if true, complete an AccuracyReport for unit tests
                  to verify that performance is adequate
  :param backprop_through_attack: If True, backprop through adversarial
                                  example construction process during
                                  adversarial training.
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))
  x_train1, y_train1 = get_train(FLAGS.train1)
  x_test1, y_test1 = get_test(FLAGS.test1)
  x_train, y_train = x_train1, y_train1
  x_test, y_test = x_test1, y_test1
  if (FLAGS.train2):
    x_train2, y_train2, x_test2, y_test2 = get_train(FLAGS.train2)
    x_train, y_train = zip_trains(x_train1, y_train1, x_train2, y_train2, 0.5)
    x_test, y_test = zip_tests(x_test1, y_test1, x_test2, y_test2, 0.5)

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))
  print(x)
  print(y)
  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  eval_params = {'batch_size': batch_size}
  rng = np.random.RandomState([2017, 8, 30])

  def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    setattr(report, report_key, acc)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

  if clean_train:
    model = ModelBasicCNN('model1', nb_classes, nb_filters)
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=label_smoothing)

    def evaluate():
      do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)

    train(sess, loss, x_train, y_train, evaluate=evaluate,
          args=train_params, rng=rng, var_list=model.get_params())
    # Calculate training error
    if testing:
      do_eval(preds, x_train, y_train, 'train_clean_train_clean_eval')

  return report


def main(argv=None):
  """
  Run the tutorial using command line flags.
  """
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 clean_train=FLAGS.clean_train,
                 backprop_through_attack=FLAGS.backprop_through_attack,
                 nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_bool('clean_train', CLEAN_TRAIN, 'Train on clean examples')
  flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))
  flags.DEFINE_string('train1', '', '')
  flags.DEFINE_string('train2', '', '')
  flags.DEFINE_string('test1', '', '')
  flags.DEFINE_string('test2', '', '')
  flags.DEFINE_string('adv', '', '')
  main()
