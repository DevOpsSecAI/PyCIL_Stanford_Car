#!/bin/sh
set -e

chmod +x train.sh install_awscli.sh

mkdir upload

python server.py
