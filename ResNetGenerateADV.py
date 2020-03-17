import keras
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from keras.datasets import cifar10
import tensorflow as tf
from keras.optimizers import Adam
from PIL import Image
import numpy as np


def get_adversarial_loss(model, fgsm, fgsm_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.get_input_at(0), **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        return 0.5 * cross_ent + 0.5 * cross_ent_adv

    return adv_loss


def rotate(points, theta):
    rotation_matrix = tf.stack([tf.cos(theta), -tf.sin(theta), 0,
                                tf.sin(theta), tf.cos(theta), 0,
                                0, 0, 1])
    rotation_matrix = tf.reshape(rotation_matrix, (3, 3))
    return tf.matmul(points, rotation_matrix)


def rotate_random(points, theta_range):
    theta = tf.random.uniform(shape=(), minval=theta_range[0], maxval=theta_range[1], dtype=tf.dtypes.float32)
    rotation_matrix = tf.stack([tf.cos(theta), -tf.sin(theta), 0,
                                tf.sin(theta), tf.cos(theta), 0,
                                0, 0, 1])
    rotation_matrix = tf.reshape(rotation_matrix, (3, 3))
    return tf.matmul(points, rotation_matrix)


def resize_and_pad_random(points, image_height, image_width, ratio_range=(0.8, 1.0)):
    new_height = tf.cast(
        tf.random.uniform(shape=(), minval=image_height * ratio_range[0], maxval=image_height * ratio_range[1]),
        dtype=tf.dtypes.int32)
    new_width = tf.cast(
        tf.random.uniform(shape=(), minval=image_width * ratio_range[0], maxval=image_width * ratio_range[1]),
        dtype=tf.dtypes.int32)
    points = tf.image.resize(points, tf.stack([new_height, new_width]))
    points = tf.image.pad_to_bounding_box(points, 32 - new_height, 32 - new_width, 32, 32)
    return points


def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.get_input_at(0), **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc


def get_adversarial_rotated_acc_metric(theta, model, fgsm, fgsm_params):
    def adv_rotated_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.get_input_at(0), **fgsm_params)
        x_adv = rotate(x_adv, theta)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    adv_rotated_acc.__name__ = 'rotated_{}'.format(theta)
    return adv_rotated_acc


def get_adversarial_random_acc_metric(model, fgsm, fgsm_params, theta_range=(-45, 45), ratio_range=(0.8, 1.0)):
    def adv_random_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.get_input_at(0), **fgsm_params)
        x_adv = rotate_random(x_adv, theta_range)
        x_adv = resize_and_pad_random(x_adv, 32, 32, ratio_range)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)
        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_random_acc


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


num_classes = 10

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model0 = keras.models.load_model('cifar_model/resnet/resnet_cifar_pure.h5')
fgsm_params = {'eps': 0.05,
               'clip_min': 0.,
               'clip_max': 1.}

from keras import backend as K
sess = K.get_session()

wrap = KerasModelWrapper(model0)
fgsm = FastGradientMethod(wrap, sess=sess)

# (32, 32, 3)
img_sample = Image.fromarray((x_test[1]*255).astype('uint8'), 'RGB')
img_sample.save('raw.bmp')
x_adv = fgsm.generate_np(x_test[1:2], **fgsm_params)[0]
img_adv_sample = Image.fromarray((x_adv*255).astype('uint8'), 'RGB')
img_adv_sample.save('adv.bmp')


"""model1 = keras.models.load_model('cifar_model/resnet/resnet_cifar_fgsm_adv_train_loss3pure7adv_puregenadv.h5',
                                 custom_objects={'adv_loss': 'categorical_crossentropy',
                                                 'adv_acc': 'accuracy'})
adv_acc = get_adversarial_acc_metric(model1, fgsm, fgsm_params)
adv_random = get_adversarial_random_acc_metric(model1, fgsm, fgsm_params, (-45, 45), (0.6, 1.0))
metrics = ['accuracy', adv_acc, adv_random]
adv_loss = get_adversarial_loss(model1, fgsm, fgsm_params)
model1.compile(
    optimizer=Adam(learning_rate=lr_schedule(0)),
    loss=adv_loss,
    metrics=metrics
)
print(model1.evaluate(x_test, y_test, verbose=1))
print(model1.metrics_names)"""

