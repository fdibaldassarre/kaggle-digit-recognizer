#!/usr/bin/env python3

import json

import numpy

import time
rng = numpy.random.RandomState(int(time.time()))

import theano
from theano import tensor as T

import lasagne
from lasagne.nonlinearities import LeakyRectify
leaky_rectify = LeakyRectify(0.1)

from PIL import Image

class CNN():
  
  def __init__(self):
    self.initializeInputs()
    self.initializeShapes()
    self.initializeTrainingSettings()
  
  def initializeTrainingSettings(self):
    self.learning_rate = 0.001
    self.momentum = 0.9
    self.max_epochs = 100
    self.minibatch_size = 350
    self.validation_error_target = 0.001
    
  def initializeNetwork(self):
    self.initializeParameters()
    self.loadLayers()
  
  def loadNetwork(self, filepath):
    self.loadModelFromFile(filepath)
    self.loadLayers()
  
  def initializeInputs(self):
    self.input = T.tensor4(name='input') # For the love of God DO NOT SET dtype= in any way
    self.output_train = T.ivector('targets')
  
  def initializeShapes(self):
    self.shapes = [0] * 4
    # shape = (input_size, output_size, convolution_size)
    self.shapes[0] = (1, 4, 5) # Conv layers
    self.shapes[1] = (4, 8, 3)
    self.shapes[2] = (200, 512, 0)  # Linear layers
    self.shapes[3] = (512, 10, 0)
    self.n_layers = len(self.shapes)
    self.n_last_conv = 1
      
  def initializeParameters(self):
    self.parameters = [0] * self.n_layers
    for i in range(self.n_layers):
      n_in, n_out, conv_size = self.shapes[i]
      b_shape = (n_out, )
      if i <= self.n_last_conv:
        w_shape = (n_out, n_in, conv_size, conv_size)
        w, b = self.randomInitParams(w_shape, b_shape)
        self.parameters[i] = (w, b)
      else:
        w_shape = (n_in, n_out)
        w, b = self.randomInitParams(w_shape, b_shape)
        self.parameters[i] = (w,b)
  
  def randomInitParams(self, w_shape, b_shape):
    # Set weights bound
    if len(w_shape) == 2:
      w_bound = numpy.sqrt(w_shape[0]*w_shape[1]) # dense layer
    else:
      w_bound = numpy.sqrt(w_shape[1]*w_shape[2]*w_shape[3]) # convolution layer
    # Initalize weights randomly with values in [-w_bound, w_bound]
    w = theano.shared(numpy.asarray(
                              rng.uniform(
                                low=-1.0 / w_bound,
                                high=1.0 / w_bound,
                                size=w_shape), dtype=self.input.dtype))
    # Initialize bias to zero
    b = theano.shared(numpy.zeros(b_shape, dtype=self.input.dtype))
    return (w, b)
      
  def loadLayers(self):
    layer = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=self.input)
    print('Input layer:', layer.output_shape)
    # Convolution
    w, b = self.parameters[0]
    _, n_out, conv_size = self.shapes[0]
    layer = lasagne.layers.Conv2DLayer(layer, n_out, (conv_size, conv_size), nonlinearity=leaky_rectify, W=w, b=b)
    print(layer.input_shape, '-->', layer.output_shape)
    # Max pooling
    layer = lasagne.layers.MaxPool2DLayer(layer, pool_size=(2, 2))
    print(layer.input_shape, '-->', layer.output_shape)
    # Convolution
    w, b = self.parameters[1]
    _, n_out, conv_size = self.shapes[1]
    layer = lasagne.layers.Conv2DLayer(layer, n_out, (conv_size, conv_size), nonlinearity=leaky_rectify, W=w, b=b)
    print(layer.input_shape, '-->', layer.output_shape)
    # Max pooling
    layer = lasagne.layers.MaxPool2DLayer(layer, pool_size=(2, 2))
    print(layer.input_shape, '-->', layer.output_shape)
    # Dense
    w, b = self.parameters[2]
    _, n_out, _ = self.shapes[2]
    layer = lasagne.layers.DenseLayer(layer, num_units=n_out, nonlinearity=leaky_rectify, W=w, b=b)
    print(layer.input_shape, '-->', layer.output_shape)
    # Dense
    w, _ = self.parameters[3]
    _, n_out, _ = self.shapes[3]
    layer = lasagne.layers.DenseLayer(layer, num_units=n_out, nonlinearity=lasagne.nonlinearities.softmax, W=w)
    print(layer.input_shape, '-->', layer.output_shape)
    # Set output
    self.network = layer
    self.output = lasagne.layers.get_output(self.network)
  
  def buildPredictFunction(self):
    return theano.function([self.input], self.output)
  
  def getPredictFunction(self):
    f = self.buildPredictFunction()
    return lambda x : f(x).argmax(axis=1)
  
  def buildTrainingFunction(self):
    # set cost_function
    cost_function = lasagne.objectives.categorical_crossentropy(self.output, self.output_train).mean()
    # compute updates
    parameters = lasagne.layers.get_all_params(self.network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(cost_function, parameters, self.learning_rate, self.momentum)
    return theano.function([self.input, self.output_train], cost_function, updates=updates)
  
  def buildValidationFunction(self):
    cost_function = lasagne.objectives.categorical_crossentropy(self.output, self.output_train).mean()
    return theano.function([self.input, self.output_train], cost_function)
  
  def train(self, data_x, data_y, savepath=None):
    # Split in train data + validation data
    n_data = len(data_x)
    n_train = int(n_data*0.80)
    n_validation = n_data - n_train
    train_x = data_x[:n_train, :, :, :]
    train_y = data_y[:n_train]
    validation_x = data_x[n_train:, :, :, :]
    validation_y = data_y[n_train:]
    # Get training and validation functions
    t = self.buildTrainingFunction()
    v = self.buildValidationFunction()
    best_validation = None
    for epoch in range(1, self.max_epochs+1):
      print('Epoch', epoch)
      # Train
      batch_number = 0
      error_train = 0
      for batch in self.iterateMiniBatch(train_x, train_y):
        batch_x, batch_y = batch
        error_batch = t(batch_x, batch_y)
        error_train += error_batch
        batch_number += 1
      training_error = error_train / batch_number
      print('Training error:', training_error)
      if epoch%5 == 0 or epoch == self.max_epochs:
        # Validate
        batch_number = 0
        error_validate = 0
        for batch in self.iterateMiniBatch(validation_x, validation_y):
          batch_x, batch_y = batch
          error_batch = t(batch_x, batch_y)
          error_validate += error_batch
          batch_number += 1
        validate_error = error_validate / batch_number
        print('Validation error:', validate_error)
        if validate_error <= self.validation_error_target:
          print('Reached validation error target, stop training')
          best_validation = validate_error
          if savepath is not None:
            print('Saving model...')
            self.saveModel(savepath)
          break
        elif best_validation is None or validate_error < best_validation:
          best_validation = validate_error
          if savepath is not None:
            print('Saving model...')
            self.saveModel(savepath)
    print('Training ended')
  
  def iterateMiniBatch(self, x, y):
    n = len(x)
    index = numpy.arange(n)
    numpy.random.shuffle(index)
    if n > self.minibatch_size:
      n_minibatches = int(n / self.minibatch_size)
      # Pick self.minibatch_size random indices
      for s in range(n_minibatches):
        indices = index[s*self.minibatch_size:(s+1)*self.minibatch_size]
        yield x[indices], y[indices]
      #remainder = n - n_minibatches*self.minibatch_size
      #if remainder > 0:
      #  indices = index[n_minibatches*self.minibatch_size:]
      #  return x[indices], y[indices]
    else:
      # Return the entire batch
      return x, y
  
  def loadData(self, filepath, test=False):
    hand = open(filepath, 'r')
    # Skip header
    hand.readline()
    data_x = []
    data_y = []
    for line in hand:
      line = line.rstrip()
      tmp = line.split(',')
      if not test:
        tmp.reverse()
        y = tmp.pop()
        tmp.reverse()
      else:
        y = 1
      x = tmp
      data_x.append(x)
      data_y.append(y)
    n = len(data_x)
    data_x = numpy.asarray(data_x, dtype=self.input.dtype).reshape(n, 1, 28, 28)/255.
    data_y = numpy.asarray(data_y, dtype='int32')
    return data_x, data_y
  
  def saveModel(self, filepath):
    # Extract parameters
    model = []
    for params in self.parameters:
      if len(params) > 1:
        ws, bs = params
        w = ws.get_value().tolist()
        b = bs.get_value().tolist()
        # get sizes
        model.append((w,b))
      else:
        ws = params[0]
        w = ws.get_value().tolist()
        model.append((w,))
    # Save to file
    encoder = json.JSONEncoder()
    data = encoder.encode(model)
    hand = open(filepath, 'w')
    hand.write(data)
    hand.close()
  
  def loadModelFromFile(self, filepath):
    # Json decoder
    decoder = json.JSONDecoder()
    # Read the model from json file
    hand = open(filepath, 'r')
    tmp = []
    for line in hand:
      line = line.strip()
      tmp.append(line)
    hand.close()
    tmp = ''.join(tmp)
    # Decode
    parameters = decoder.decode(tmp)
    self.loadModel(parameters)
  
  def loadModel(self, parameters):
    self.parameters = []
    for params in parameters:
      if len(params) == 1:
        w = params[0]
        wt = theano.shared(numpy.asarray(w, dtype=self.input.dtype))
        self.parameters.append((wt,))
      else:
        w,b = params
        wt = theano.shared(numpy.asarray(w, dtype=self.input.dtype))
        bt = theano.shared(numpy.asarray(b, dtype=self.input.dtype))
        self.parameters.append((wt,bt))
  
  def writeSubmission(self, data_y, savepath):
    with open(savepath, 'w') as hand:
      hand.write('ImageId,Label' + '\n')
      for i in range(len(data_y)):
        hand.write(str(i+1) + ',' + str(data_y[i]) + '\n')
  
  def convertImageToData(self, imgpath):
    img = Image.open(imgpath)
    img = img.convert('L').resize((28,28))
    data = numpy.asarray(img, dtype=numpy.float32) / 255.
    return data.reshape(1,1,28,28)
