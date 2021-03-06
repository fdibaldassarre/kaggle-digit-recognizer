#!/usr/bin/env python3

import os
import time

from src.Constants import MODELS_FOLDER
from src.Constants import DATA_FOLDER

MODEL_FILE = os.path.join(MODELS_FOLDER, 'model.json')
DATA_FILE = os.path.join(DATA_FOLDER, 'train.csv')

from src.CNN import CNN

cnn = CNN()
if os.path.exists(MODEL_FILE):
  print('Loading model...')
  cnn.loadNetwork(MODEL_FILE)
else:
  print('Initialize model...')
  cnn.initializeNetwork()
# Load data
print('Loading data...')
data_x, data_y = cnn.loadData(DATA_FILE)
# Add more data
data_x, data_y = cnn.increaseData(data_x, data_y)
# Shuffle
data_x, data_y = cnn.shuffleData(data_x, data_y)
# Shape for training
data_x, data_y = cnn.reshapeData(data_x, data_y)
# Train
print('Training...')
start_time = time.time()
validation_data, _ = cnn.train(data_x, data_y, savepath=MODEL_FILE)
end_time = time.time()
time_train = round((end_time - start_time) / 60)
print('Training took', time_train, 'minutes')
# Reload best model
print('Reload best model up to now')
self.loadModel(MODEL_FILE)
# Get validation success rate
print('Get success rate')
valid_x, valid_y = validation_data
success_rate = cnn.getSuccessRate(valid_x, valid_y)
print('Success rate on validation:', success_rate, '%')
