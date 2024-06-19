#!/bin/sh
kaggle datasets download -d senemanu/stanfordcarsfcs

unzip -qq stanfordcarsfcs.zip

rm -rf ./car_data/car_data/train/models

mv ./car_data/car_data/test ./car_data/car_data/val
