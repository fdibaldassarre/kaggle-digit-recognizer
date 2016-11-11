#!/usr/bin/env python3

import os

path = os.path.abspath(__file__)
MAIN_FOLDER = os.path.dirname(path)
MODELS_FOLDER = os.path.join(MAIN_FOLDER, 'models/')
DATA_FOLDER = os.path.join(MAIN_FOLDER, 'data/')

MODEL_FILE = os.path.join(MODELS_FOLDER, 'model.json')
TRAIN_FILE = os.path.join(DATA_FOLDER, 'train.csv')
DATA_FILE = os.path.join(DATA_FOLDER, 'test.csv')
SUBMISSION_FILE = os.path.join(MAIN_FOLDER, 'submission.csv')

from src.CNN import CNN

cnn = CNN()
if os.path.exists(MODEL_FILE):
  cnn.loadNetwork(MODEL_FILE)
else:
  print('WARNING! No model file found. Creating new network.')
  cnn.initializeNetwork()

data_x, data_y = cnn.loadData(TRAIN_FILE)
f = cnn.getPredictFunction()
predict_y = f(data_x)
right = 0
n = len(data_y)
for i in range(n):
  if predict_y[i] == data_y[i]:
    right += 1

percentage = 100. * right / n
print('Success rate on training data: ', percentage, '%')

print('Write submission')
data_x, _ = cnn.loadData(DATA_FILE, test=True)
data_y = f(data_x)
cnn.writeSubmission(data_y, SUBMISSION_FILE)
