import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}_{}/{}/{}".format(args["model_name"],args["dataset"], args['data'], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    save_name = "models/{}/{}_{}/{}/{}".format(args["model_name"],args["dataset"], args['data'], init_cls, args['increment'])
    
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    logfilename = "logs/{}/{}_{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        args['data'],
        init_cls,
        args["increment"],
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
        force=True
    )
    args['logfilename'] = logs_name
    args['csv_name'] = "{}_{}_{}".format(
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )


    _set_random()
    _set_device(args)
    print_args(args)
    model = factory.get_model(args["model_name"], args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        path = args["data"],
    )
    if data_manager.get_task_size(0) < 5:
        top_string = "top{}".format(data_manager.get_task_size(0))
    else:
        top_string = "top5"
    cnn_curve, nme_curve = {"top1": [], top_string: []}, {"top1": [], top_string: []}
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager.nb_tasks):
        print(args["device"])
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task(save_conf=True)
        model.after_task()

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
            cnn_curve[top_string].append(cnn_accy["top{}".format(model.topk)])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve[top_string].append(nme_accy["top{}".format(model.topk)])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve[top_string]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve[top_string]))

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
            cnn_curve[top_string].append(cnn_accy["top{}".format(model.topk)])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve[top_string]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
        model.save_checkpoint(save_name)
        if len(cnn_matrix)>0:
            np_acctable = np.zeros([  task + 1, int((args["init_cls"] // 10) + task * (args["increment"] // 10))])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, -1])[:-1])
            logging.info('Forgetting (CNN): {}'.format(forgetting))
            logging.info('Accuracy Matrix (CNN): {}'.format(np_acctable))
            print('Accuracy Matrix (CNN):')
            print(np_acctable)
            print('Forgetting (CNN):', forgetting)
        if len(nme_matrix)>0:
            np_acctable = np.zeros([ task + 1, int((args["init_cls"] // 10) + task * (args["increment"] // 10))])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, -1])[:-1])
            logging.info('Forgetting (NME): {}'.format(forgetting))
            logging.info('Accuracy Matrix (NME): {}'.format(np_acctable))
            print('Accuracy Matrix (NME):')
            print(np_acctable)
            print('Forgetting (NME):', forgetting)

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


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

