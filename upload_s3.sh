#!/bin/sh

# Ensure the script exits on the first error and prints each command before executing it
set -e
set -x

# Check if local directory and s3 path arguments were provided, if not, set default values
local_dir=${1:-./models}
s3_path=${2:-s3://pycil.com/"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"}

# Perform the S3 copy operation with the provided or default s3 path
aws s3 cp "$local_dir" "$s3_path" --recursive
