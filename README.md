# Digit Recognizer

Digit recognizer made with Theano+Lasagne.

## Installation requirements

- Python 3
- Theano
- Lasagne
- PIL

## Get the data

Put the train.csv and test.csv files from [Kaggle-Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data) into
the data folder.

## Usage

Run
```sh
./train.py
```
to create the model and run
```sh
./test.py
```
to write the submission file.

In my case the training took 37 minutes on an Intel i5-6200U and the model had a 98.9% success rate on the validation set.

Use
```sh
./recognize.py /path/to/image
```
to recognize the digit in an image (see the test folder for sample images).
