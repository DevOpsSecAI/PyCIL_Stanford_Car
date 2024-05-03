# !bin/bash
kaggle datasets download -d senemanu/stanfordcarsfcs

upzip stanfordcarsfcs.zip

rm -rf ./car_data/car_data/train/models