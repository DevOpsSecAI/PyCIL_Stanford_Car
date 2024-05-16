#!/bin/sh

# Run the long-running train.sh script in the background
./train.sh ./exps/simplecil.json &

# Run the entrypoint.sh script
date 

# Wait for all background processes to complete
wait
