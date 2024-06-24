#!/bin/sh

# Ensure the script exits on the first error and prints each command before executing it
set -e
set -x

# Check if config, data, upload_s3_arg, and s3_path arguments were provided, if not, set default values
config=${1:-./exps/simplecil_general.json}
data=${2:-./car_data/car_data}
upload_s3_arg=${3:-./models}
s3_path=${4:-s3://pycil.com/"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"}

# Run the training script with the provided or default config and data arguments
python main.py --config "$config" --data "$data"

# Check if the previous command was successful
if [ $? -eq 0 ]; then
    # Run the upload script with the additional arguments
    ./upload_s3.sh "$upload_s3_arg" "$s3_path"
else
    echo "Error: python main.py failed. Aborting upload."
    exit 1
fi
