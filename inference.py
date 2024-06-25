import sys
import logging
import copy
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import pil_to_tensor
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from utils.data_manager import pil_loader
import os
import numpy as np
import json
import argparse
import imghdr
import time

def is_image_imghdr(path):
  """
  Checks if a path points to a valid image using imghdr.

  Args:
      path: The path to the file.

  Returns:
      True if the path is a valid image, False otherwise.
  """
  if not os.path.isfile(path):
      return False
  return imghdr.what(path) in ['jpeg', 'png']

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
def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    assert args['output'].split(".")[-1] == "json" or os.path.isdir(args['output'])
    model = load_model(args)
    result = []
    if is_image_imghdr(args['input']):
        img = pil_to_tensor(pil_loader(args['input']))
        img = img.unsqueeze(0)
        predictions = model.inference(img)
        out = {"img": args['input'].split("/")[-1]}
        out.update({"predictions": [{"confident": confident, "index": pred, "label": label } for pred, label, confident in zip(predictions[0], predictions[1], predictions[2])]})
        result.append(out)
    else:
        image_list = filter(lambda x: is_image_imghdr(os.path.join(args['input'], x)), os.listdir(args['input']))
        for image in image_list:
            print("Inference on image", image)
            img = pil_to_tensor(pil_loader(os.path.join(args['input'], image)))
            img = img.unsqueeze(0)
            predictions = model.inference(img)
            out = {"img": image.split("/")[-1]}
            out.update({"predictions": [{"confident": confident, "index": pred, "label": label } for pred, label, confident in zip(predictions[0], predictions[1], predictions[2])]})
            result.append(out)
    if args['output'].split(".")[-1] == "json":
        with open(args['output'], "w+") as f:
            json.dump(result, f, indent=4)
    else:
        with open(os.path.join(args['output'], "output_model_{}.json".format(time.time())), "w+") as f:
            json.dump(result, f, indent=4)
def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, help='Json file of settings.')
    parser.add_argument('--checkpoint', type=str, help="path to checkpoint file. File must be a .pth format file")
    parser.add_argument('--input', type=str, help="Path to input. This could be an folder or an image file")
    parser.add_argument('--output', type=str, help = "Output path to save prediction")
    return parser
    
if __name__ == '__main__':
    main()

