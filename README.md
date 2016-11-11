# Digit Recognizer

Simple digit recognizer made with Theano+Lasagne.

## Installation requirements

- Python 3
- Theano
- Lasagne

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
to print the success rate on the training set and write the submission file.

The training takes 20-30 minutes on an Intel 6100U cpu and the result model has a 98% success rate.
