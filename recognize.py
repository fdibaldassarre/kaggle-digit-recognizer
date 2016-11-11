#!/usr/bin/env python3

import os
import sys

path = os.path.abspath(__file__)
MAIN_FOLDER = os.path.dirname(path)
MODELS_FOLDER = os.path.join(MAIN_FOLDER, 'models/')
MODEL_FILE = os.path.join(MODELS_FOLDER, 'model.json')

import argparse

from src.CNN import CNN

parser = argparse.ArgumentParser(description="Recognize Digit")
parser.add_argument('input', help='Input image')
parser.add_argument('--model', '-m', dest='model', default=None,
                    help='Model file')
args = parser.parse_args()

imgpath = args.input
model_path = args.model

model_path = MODEL_FILE if model_path is None else model_path

# Check inputs
if not os.path.exists(imgpath):
  print('Input image not found.')
  sys.exit(1)
if not os.path.exists(model_path):
  print('Model file does not exist.')
  sys.exit(1)

# Load network
cnn = CNN()
cnn.loadNetwork(model_path)
# Convert image to data
data_x = cnn.convertImageToData(imgpath)
# Predict
f = cnn.getPredictFunction()
data_y = f(data_x)[0]
print('Digit:', data_y)
