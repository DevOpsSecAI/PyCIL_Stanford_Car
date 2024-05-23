#!/bin/sh

# Ensure the script exits on the first error and prints each command before executing it
set -e
set -x

# Perform the S3 copy operation with a timestamp in the destination path
aws s3 cp ./models/simplecil s3://pycil.com/"$(date -u +"%Y-%m-%dT%H:%M:%SZ")" --recursive
