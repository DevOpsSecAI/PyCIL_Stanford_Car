#!/bin/sh
set -e

chmod +x train.sh install_awscli.sh

python server.py
