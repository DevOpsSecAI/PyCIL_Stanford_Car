import json
import argparse
from trainer import train
from train_more import train_more

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    if not args['dataset'] == "general_dataset":    
        train(args)
    else:
        assert args['data'] != None
        if not args['checkpoint']:
            train(args)
        else:
            train_more(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('-d','--data', nargs ='?', type=str, help='Path of the data folder')
    parser.add_argument('-c','--checkpoint',nargs = '?', type=str, help='Path of checkpoint file if resume training')
    return parser


if __name__ == "__main__":
    main()
