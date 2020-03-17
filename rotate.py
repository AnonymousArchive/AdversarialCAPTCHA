import os
import numpy as np
from PIL import Image
import uuid

def data_disk(data_dir, train_start=0,
               train_end=6000, test_start=0, test_end=1000):
  assert isinstance(train_start, int)
  assert isinstance(train_end, int)
  assert isinstance(test_start, int)
  assert isinstance(test_end, int)

  X_train, Y_train = parse_disk_file(data_dir + '/train')
  X_test, Y_test = parse_disk_file(data_dir + '/test')
  return X_train, Y_train, X_test, Y_test

def parse_disk_file(data_dir):
    assert os.path.exists(data_dir), data_dir
    filenames = [filename for filename in os.listdir(data_dir) if filename.endswith('.png')]
    dim_sizes = (len(filenames), 28, 28, 1)
    y = np.zeros(shape=(len(filenames), 10))
    x = np.zeros(shape=dim_sizes)
    for index in range(len(filenames)):
      filename = filenames[index]
      label = int(filename.split('_')[0])
      label_array = np.zeros(shape=(10))
      label_array[label] = 1
      y[index] = label_array
      raw_image = np.asarray(Image.open(data_dir+'/'+filename)).reshape((28, 28, 1))
      x[index] = raw_image
    return x, y


def rotate(atk, degree):
    in_path = 'images/' + atk + '_adv'
    x_train, y_train, x_test, y_test = data_disk(in_path, train_start=0, train_end=60000, test_start=0,
                                                 test_end=10000)

    dir = 'images/' + atk + '_rotated' + str(degree) + '_adv/'
    if not os.path.exists('images'):
        os.mkdir('images')
    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(dir + 'train/'):
        os.mkdir(dir + 'train/')
    if not os.path.exists(dir + 'test/'):
        os.mkdir(dir + 'test/')
    for index in range(len(y_test)):
        if index % 5000 == 0:
            print('test ' + str(index))
        x_ = x_test[index]
        label = np.argmax(y_test[index])
        raw_data = x_.reshape((28, 28)).astype('uint8')
        im = Image.fromarray(raw_data, mode='P')
        rot = im.rotate(degree)
        rot.save(dir + 'test/' + str(label) + '_' + str(uuid.uuid4()) + '.png')

    for index in range(len(y_train)):
        if index % 5000 == 0:
            print('train ' + str(index))
        x_ = x_train[index]
        label = np.argmax(y_train[index])
        raw_data = x_.reshape((28, 28)).astype('uint8')
        im = Image.fromarray(raw_data, mode='P')
        rot = im.rotate(degree)
        rot.save(dir + 'train/' + str(label) + '_' + str(uuid.uuid4()) + '.png')

degrees = [ i for i in range(5, 46, 5) ]
atks = ['fgsm', 'jsma']

pairs = []

for atk in atks:
    for degree in degrees:
        pairs.append((atk, degree))

for pair in pairs:
    print(pair)
    rotate(pair[0], pair[1])