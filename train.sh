#!/bin/sh
for arg in $@; do
  python ./main.py --config=$arg;
  # Your commands to process each argument here
done