#!/usr/bin/env python3

import gzip
import numpy as np
from PIL import Image

def sigmoid(x):
  return 1 / (1+np.exp(-x))

def relu(x):
  return np.maximum(0, x)

def softmax(x):
  c = np.max(x)
  return np.exp(x-c) / np.sum(np.exp(x-c))


class Mnist:
  def __init__(self):
    self.image_size = (28*28)
    self.dataset_dict = dict()

    with gzip.open('./t10k-images-idx3-ubyte.gz', 'rb') as f:
      raw_data = np.frombuffer(f.read(), np.uint8, offset=16)
    self.dataset_dict['test_image'] = raw_data.reshape(-1,self.image_size)

    with gzip.open('./train-images-idx3-ubyte.gz', 'rb') as f:
      raw_data = np.frombuffer(f.read(), np.uint8, offset=16)
    self.dataset_dict['train_image'] = raw_data.reshape(-1,self.image_size)

    with gzip.open('./t10k-labels-idx1-ubyte.gz', 'rb') as f:
      raw_data = np.frombuffer(f.read(), np.uint8, offset=8)
    self.dataset_dict['test_label'] = raw_data

    with gzip.open('./train-labels-idx1-ubyte.gz', 'rb') as f:
      raw_data = np.frombuffer(f.read(), np.uint8, offset=8)
    self.dataset_dict['train_label'] = raw_data

    """
    data = self.test_image_data_matrix[100:101]
    data2 = data.reshape(-1,28)
    pil_img = Image.fromarray(data2)
    pil_img.save('10.png')
    """

  def to_one_hot_label(self, array):
    T = np.zeros((array.size, 10))
    for idx, row in enumerate(T):
      num = array[idx]
      row[num] = 1
    return T

  def load(self, normalize=True, flatten=True, onehot=True):

    if normalize == True:
      for key in ['test_image', 'train_image']:
        self.dataset_dict[key] = self.dataset_dict[key].astype(np.float32)
        self.dataset_dict[key] /= 255.0

    if onehot == True:
      self.dataset_dict['test_label'] = self.to_one_hot_label(self.dataset_dict['test_label']) 
      self.dataset_dict['train_label'] = self.to_one_hot_label(self.dataset_dict['train_label']) 
    return self.dataset_dict['test_image'], self.dataset_dict['test_label'], self.dataset_dict['train_image'], self.dataset_dict['train_label'],

class Neural:
  def __init__(self, img, label):
    self.img = img
    self.label = label
    self.weight = dict()
    self.weight['w1'] = np.random.randn(784, 50)
    self.weight['w2'] = np.random.randn(50, 100)
    self.weight['w3'] = np.random.randn(100, 10)
    self.weight['b1'] = np.random.randn(50)
    self.weight['b2'] = np.random.randn(100)
    self.weight['b3'] = np.random.randn(10)

  def predict(self):
    w1 = self.weight['w1']
    w2 = self.weight['w2']
    w3 = self.weight['w3']
    b1 = self.weight['b1']
    b2 = self.weight['b2']
    b3 = self.weight['b3']

    match_count = 0
    for x, label in zip(self.img, self.label):
      ret1 = np.dot(x, w1) + b1
      a1 = sigmoid(ret1)
      ret2 = np.dot(a1, w2) + b2
      a2 = sigmoid(ret2)
      ret3 = np.dot(a2, w3) + b3
      a3 = sigmoid(ret3)
      y = softmax(a3)
      result_index = np.argmax(y)
      if result_index == label:
        match_count += 1
    print(f'{match_count}/{len(self.img)}, {match_count/len(self.img)*100:.3f}')

if __name__ == '__main__':
  mnist = Mnist()
  test_img, test_label, train_img, train_label = mnist.load(normalize=True, flatten=True, onehot=False)
  neural = Neural(test_img, test_label)
  neural.predict()





