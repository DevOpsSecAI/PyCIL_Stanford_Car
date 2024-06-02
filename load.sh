#! /bin/sh
for arg in $@; do
  python ./load_model.py --config=$arg;
  # Your commands to process each argument here
done