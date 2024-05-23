#!/bin/sh
kaggle datasets download -d senemanu/stanfordcarsfcs

unzip -q stanfordcarsfcs.zip

rm -rf ./car_data/car_data/train/models
