#!/bin/bash

python salt_identification_2.py --optimizer sgd --batch_size 4 --nb_epochs 120
python salt_identification_2.py --optimizer sgd --batch_size 8 --nb_epochs 120
python salt_identification_2.py --optimizer sgd --batch_size 16 --nb_epochs 120