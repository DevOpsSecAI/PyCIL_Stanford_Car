import json
import argparse
import os
from trainer import train

from logtail import LogtailHandler
import logging

handler = LogtailHandler(source_token=os.environ['SOURCE_TOKEN'])
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = []
logger.addHandler(handler)


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')

    return parser


if __name__ == '__main__':
    main()
