#!/bin/bash

python salt_identification.py --optimizer sgd --decay 1e-3 --lr 0.01
python salt_identification.py --optimizer sgd --decay 1e-4 --lr 0.1
python salt_identification.py --optimizer sgd --decay 1e-3 --lr 0.1