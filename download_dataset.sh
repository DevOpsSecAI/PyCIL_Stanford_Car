# !bin/bash
kaggle datasets download -d senemanu/stanfordcarsfcs

unzip stanfordcarsfcs.zip

rm -rf ./car_data/car_data/train/models