#!/usr/bin/env python3

import os
import time

path = os.path.abspath(__file__)
MAIN_FOLDER = os.path.dirname(path)
MODELS_FOLDER = os.path.join(MAIN_FOLDER, 'models/')
DATA_FOLDER = os.path.join(MAIN_FOLDER, 'data/')

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
#data_x, data_y = cnn.increaseData(data_x, data_y)
# Shuffle
data_x, data_y = cnn.shuffleData(data_x, data_y)
# Shape for training
data_x, data_y = cnn.reshapeData(data_x, data_y)
# Train
print('Training...')
start_time = time.time()
valid_x, valid_y = cnn.train(data_x, data_y, savepath=MODEL_FILE)
end_time = time.time()
time_train = round((end_time - start_time) / 60)
print('Training took', time_train, 'minutes')
# Get validation success rate
success_rate = cnn.getSuccessRate(valid_x, valid_y)
print('Success rate on validation:', success_rate, '%')
