#!/bin/bash

python salt_identification.py --optimizer adam --batch_size 32 --nb_epochs 120
python salt_identification.py --optimizer adam --batch_size 32 --nb_epochs 120 --dropout 0.5
python salt_identification.py --optimizer adam --batch_size 32 --nb_epochs 120 --pooltype max
python salt_identification.py --optimizer adam --batch_size 32 --nb_epochs 120 --pooltype max --dropout 0.5
python salt_identification.py --optimizer adam --batch_size 32 --nb_epochs 120 --upsample 1
python salt_identification.py --optimizer adam --batch_size 32 --nb_epochs 120 --upsample 1 --dropout 0.5
python salt_identification.py --optimizer adam --batch_size 32 --nb_epochs 120 --upsample 1 --pooltype max
python salt_identification.py --optimizer adam --batch_size 32 --nb_epochs 120 --upsample 1 --pooltype max --dropout 0.5