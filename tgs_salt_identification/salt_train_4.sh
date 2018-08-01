#!/bin/bash

python salt_identification.py --optimizer adadelta --batch_size 4 --nb_epochs 120
python salt_identification.py --optimizer adadelta --batch_size 8 --nb_epochs 120
python salt_identification.py --optimizer adam --batch_size 4 --nb_epochs 120
python salt_identification.py --optimizer adam --batch_size 8 --nb_epochs 120