#!/bin/sh

nohup ./train.sh ./exps/simplecil.json &

echo "Training script started in the background."

# Keep the container running
echo "Container is running. Training script is executing in the background."
while :; do
    sleep 60
done
