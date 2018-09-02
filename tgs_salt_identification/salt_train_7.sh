#!/bin/bash

python salt_identification.py --optimizer adam --batch_size 32 --nb_epochs 120 --dropout 0.5
python salt_identification.py --optimizer sgd --batch_size 16 --nb_epochs 120
python salt_identification.py --optimizer sgd --batch_size 32 --nb_epochs 120