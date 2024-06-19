#!/bin/sh

python main.py --config ./exps/simplecil_general.json --data ./car_data/car_data

./upload_s3.sh
