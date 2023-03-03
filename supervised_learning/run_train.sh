#!/bin/bash
rm -rf $1;
mkdir $1;
python3 train_models.py $1 > $1/out.txt 2> $1/err.txt;