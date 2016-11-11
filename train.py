#!/usr/bin/env python3

import os

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
  cnn.initializeNetwork()

data_x, data_y = cnn.loadData(DATA_FILE)
cnn.train(data_x, data_y, savepath=MODEL_FILE)
