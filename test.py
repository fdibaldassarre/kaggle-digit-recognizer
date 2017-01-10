#!/usr/bin/env python3

import os
import sys

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
  print('WARNING! No model file found.')
  sys.exit(1)

print('Write submission')
data_x, data_y = cnn.loadData(DATA_FILE, test=True)
data_x, _ = cnn.reshapeData(data_x, data_y)
f = cnn.getPredictFunction()
data_y = f(data_x)
cnn.writeSubmission(data_y, SUBMISSION_FILE)
