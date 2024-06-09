#! /bin/sh
for arg in $@; do
  python ./main.py --config=$arg --resume;
  # Your commands to process each argument here
done