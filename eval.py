import sys
import logging
import copy
import torch
import os
from utils import factory
from utils.data_manager import DataManager
from torch.utils.data import DataLoader
from utils.toolkit import count_parameters
import os
import numpy as np
import json
import argparse
from trainer import train
import torch.nn as nn

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    
    seed_list = copy.deepcopy(args["seed"])

    for seed in seed_list:
        args["seed"] = seed
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
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
    data_manager = DataManager(
      args["dataset"],
      args["shuffle"],
      args["seed"],
      args["init_cls"],
      args["increment"],
    )
    _set_random()
    _set_device(args)
    model = factory.get_model(args["model_name"], args)
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    task_num = 0
    print(args['seed'])
    for task in range(data_manager.nb_tasks):
        task_num +=  data_manager.get_task_size(
            task
        )
        checkpoint = torch.load(os.path.join(args['checkpoint'], "{}_{}.pkl".format(args['model_name'], task)))

        model._total_classes = task_num
        model._network.update_fc(model._total_classes)
        model._network_module_ptr = model._network

        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model._network.load_state_dict(checkpoint["model_state_dict"])
        test_dataset = data_manager.get_dataset(
        np.arange(0, args["init_cls"] + task * args["increment"]), source="test", mode="test"
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4
        )
        model._network.to(args['device'][0])
        cnn_accy, nme_accy = model.eval_task(test_loader)
        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))
    
            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)
    
            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_keys_sorted = sorted(nme_keys)
            nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
            nme_matrix.append(nme_values)
    
    
            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])
    
            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])
    
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
    
            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))
    
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
    
            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)
    
            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])
    
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))
    
            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
        if len(cnn_matrix)>0:
            np_acctable = np.zeros([ task + 1, int((args["init_cls"] // 10) + task * (args["increment"] // 10))])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, -1])[:-1])
            print('Accuracy Matrix (CNN):')
            print(np_acctable)
            print('Forgetting (CNN):', forgetting)
            logging.info('Forgetting (CNN): {}'.format(forgetting))
        if len(nme_matrix)>0:
            np_acctable = np.zeros([task + 1, (args["init_cls"] // 10) + task * (args["increment"] // 10)])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, -1])[:-1])
            print('Accuracy Matrix (NME):')
            print(np_acctable)
            print('Forgetting (NME):', forgetting)
            logging.info('Forgetting (NME):', forgetting)
def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--checkpoint', type=str, default='', help="directory save model.")
    return parser
    
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

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus
    
if __name__ == '__main__':
    main()