#!/bin/bash

python salt_identification.py --optimizer adam
python salt_identification.py --optimizer adadelta
python salt_identification.py --optimizer nadam
python salt_identification.py --optimizer rmsprop
python salt_identification.py --optimizer adagrad
python salt_identification.py --optimizer adamax
python salt_identification.py --optimizer sgd