import json
import argparse
import os
from trainer import train
import logging
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

logging.basicConfig(level=logging.INFO)

sentry_sdk.init(
    dsn="https://0dbd8387b1c5f375af0fe420ea5f15f2@o4507264673710080.ingest.us.sentry.io/4507264677969920",
    integrations=[
        LoggingIntegration(
            level=logging.DEBUG,        # Capture info and above as breadcrumbs
            event_level=logging.DEBUG   # Send records as events
        ),
    ],
)


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
