import sys
import logging
import copy
import torch
from PIL import Image
import torchvision.transforms as transforms
from utils import factory
from utils.data_manager import DataManager
from torch.utils.data import DataLoader
from utils.toolkit import count_parameters
import os
import numpy as np
import json
import argparse

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus

def get_methods(object, spacing=20):
  methodList = []
  for method_name in dir(object):
    try:
        if callable(getattr(object, method_name)):
            methodList.append(str(method_name))
    except Exception:
        methodList.append(str(method_name))
  processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
  for method in methodList:
    try:
        print(str(method.ljust(spacing)) + ' ' +
              processFunc(str(getattr(object, method).__doc__)[0:90]))
    except Exception:
        print(method.ljust(spacing) + ' ' + ' getattr() failed')

def load_model(args):
    _set_device(args)
    model = factory.get_model(args["model_name"], args)
    model.load_checkpoint(args["checkpoint"])
    return model
    
def evaluate(args):
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"] + args['data'], args['init_cls'], args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        args['init_cls'],
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    args['logfilename'] = logs_name
    args['csv_name'] = "{}_{}_{}".format(
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    _set_random()
    print_args(args)
    model = load_model(args)
    
    data_manager = DataManager(
        args["dataset"],
        False,
        args["seed"],
        args["init_cls"],
        args["increment"],
        path = args["data"]
    )
    loader = DataLoader(data_manager.get_dataset(model.class_list, source = "test", mode = "test"), batch_size=args['batch_size'], shuffle=True, num_workers=8)
    
    cnn_acc, nme_acc = model.eval_task(loader, group = 1)
    print(cnn_acc, nme_acc)
def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    evaluate(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param

def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('-d','--data', type=str, help='Path of the data folder')
    parser.add_argument('-c','--checkpoint', type=str, help='Path of checkpoint file if resume training')
    return parser

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
if __name__ == '__main__':
    main()

