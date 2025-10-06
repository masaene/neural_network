#!/usr/bin/env python3

import os
import gzip
import time
import numpy as np
import common
import pickle
import torch
from PIL import Image

class Mnist:
  def __init__(self):
    self.image_size = (28*28)
    self.dataset_dict = dict()

    for load_filename, bin_name, key in [('t10k-images-idx3-ubyte.gz','./test_image.bin','test_image'), ('./train-images-idx3-ubyte.gz','./train_image.bin','train_image')]:
      if os.path.exists(bin_name):
        with open(bin_name, 'rb') as f:
          self.dataset_dict[key] = pickle.load(f)
      else:
        with gzip.open(load_filename, 'rb') as f:
          raw_data = np.frombuffer(f.read(), np.uint8, offset=16)
        self.dataset_dict[key] = raw_data.reshape(-1,self.image_size)
        with open(bin_name, 'wb') as f:
          pickle.dump(self.dataset_dict[key], f)

    for load_filename, bin_name, key in [('t10k-labels-idx1-ubyte.gz','./test_label.bin','test_label'), ('./train-labels-idx1-ubyte.gz','./train_label.bin','train_label')]:
      if os.path.exists(bin_name):
        with open(bin_name, 'rb') as f:
          self.dataset_dict[key] = pickle.load(f)
      else:
        with gzip.open(load_filename, 'rb') as f:
          raw_data = np.frombuffer(f.read(), np.uint8, offset=8)
        self.dataset_dict[key] = raw_data
        with open(bin_name, 'wb') as f:
          pickle.dump(self.dataset_dict[key], f)


    for key in ['test_image', 'train_image']:
      self.dataset_dict[key] = self.dataset_dict[key].astype(np.float32)
      self.dataset_dict[key] /= 255.0

    self.dataset_dict['train_label'] = self.to_one_hot_label(self.dataset_dict['train_label']) 

    """
    data = self.test_image_data_matrix[100:101]
    data2 = data.reshape(-1,28)
    pil_img = Image.fromarray(data2)
    pil_img.save('10.png')
    """

  def get_test_data(self):
    return self.dataset_dict['test_image'], self.dataset_dict['test_label']

  def get_train_data(self):
    return self.dataset_dict['train_image'], self.dataset_dict['train_label']

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
  def __init__(self, test_img, test_label, train_img, train_label):
    self.lr = 0.1
    self.test_img = test_img
    self.test_label = test_label
    self.train_img = train_img
    self.train_label = train_label
    self.params = dict()
    self.params['w1'] = np.random.randn(784, 50).astype(np.float32)
    self.params['w2'] = np.random.randn(50, 100).astype(np.float32)
    self.params['w3'] = np.random.randn(100, 10).astype(np.float32)
    self.params['b1'] = np.random.randn(50).astype(np.float32)
    self.params['b2'] = np.random.randn(100).astype(np.float32)
    self.params['b3'] = np.random.randn(10).astype(np.float32)
    #self.match_count = 0
    #self.total_count = 0

  def training(self):
    loss_f = lambda:self.predict_train(self.train_img, self.train_label)
    gradient_dict = dict()
    for x, label in zip(self.train_img, self.train_label):
      for key in ['w1','w2','w3','b1','b2','b3']:
        start = time.perf_counter()
        gradient[key] = common.partial_derivative(loss_f, self.params[key])
        end = time.perf_counter()
        print(f'elapsed1={end-start:.3f}ms')
      for key in ['w1','w2','w3','b1','b2','b3']:
        self.params[key] -= lr * gradient[key]

  def predict_train(self, x, l):
    w1 = self.params['w1']
    w2 = self.params['w2']
    w3 = self.params['w3']
    b1 = self.params['b1']
    b2 = self.params['b2']
    b3 = self.params['b3']

    start = time.perf_counter()
    
    tensor_x = torch.from_numpy(x)
    tensor_w1 = torch.from_numpy(w1)
    tensor_w2 = torch.from_numpy(w2)
    tensor_w3 = torch.from_numpy(w3)
    tensor_b1 = torch.from_numpy(b1)
    tensor_b2 = torch.from_numpy(b2)
    tensor_b3 = torch.from_numpy(b3)

    if torch.cuda.is_available():
      cuda_x = tensor_x.to('cuda')
      cuda_w1 = tensor_w1.to('cuda')
      cuda_w2 = tensor_w2.to('cuda')
      cuda_w3 = tensor_w3.to('cuda')
      cuda_b1 = tensor_b1.to('cuda')
      cuda_b2 = tensor_b2.to('cuda')
      cuda_b3 = tensor_b3.to('cuda')
    else:
      print("no cuda...")

    ret1 = torch.matmul(cuda_x, cuda_w1) + cuda_b1
    #a1 = ret1.sigmoid()
    a1 = torch.sigmoid(ret1)
    ret2 = torch.matmul(a1, cuda_w2) + cuda_b2
    a2 = torch.sigmoid(ret2)
    ret3 = torch.matmul(a2, cuda_w3) + cuda_b3
    a3 = torch.sigmoid(ret3)
    y_tensor = a3.softmax(dim=1).cpu()
    y = y_tensor.numpy()

    """
    ret1 = np.dot(x, w1) + b1

    a1 = common.sigmoid(ret1)
    ret2 = np.dot(a1, w2) + b2
    a2 = common.sigmoid(ret2)
    ret3 = np.dot(a2, w3) + b3
    a3 = common.sigmoid(ret3)
    y = common.softmax(a3)
    """

    e = common.cross_entropy_error(l,y) 
    end = time.perf_counter()
    #print(f'elapsed1={end-start:.3f}ms')

    return e


  def predict_test(self):
    w1 = self.params['w1']
    w2 = self.params['w2']
    w3 = self.params['w3']
    b1 = self.params['b1']
    b2 = self.params['b2']
    b3 = self.params['b3']

    total_count = 0
    match_count = 0
    for x, label in zip(self.test_img, self.test_label):
      ret1 = np.dot(x, w1) + b1
      a1 = common.sigmoid(ret1)
      ret2 = np.dot(a1, w2) + b2
      a2 = common.sigmoid(ret2)
      ret3 = np.dot(a2, w3) + b3
      a3 = common.sigmoid(ret3)
      y = common.softmax(a3)
      result_index = np.argmax(y)
      if result_index == label:
        match_count += 1
      total_count = len(self.test_img)

    print(f'{match_count}/{total_count}, {match_count/total_count*100:.3f}')

if __name__ == '__main__':
  mnist = Mnist()
  #test_img, test_label, train_img, train_label = mnist.load(normalize=True, flatten=True, onehot=True)
  test_img, test_label = mnist.get_test_data()
  train_img, train_label = mnist.get_train_data()


  neural = Neural(test_img, test_label, train_img, train_label)
  neural.training()
  neural.predict_test()

