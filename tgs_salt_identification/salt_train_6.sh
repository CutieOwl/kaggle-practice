#!/bin/bash

python salt_identification_2.py --optimizer adam --batch_size 32 --nb_epochs 120
python salt_identification.py --optimizer adam --batch_size 32 --nb_epochs 120