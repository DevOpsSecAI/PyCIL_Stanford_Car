'''
Re-implementation of SimpleCIL (https://arxiv.org/abs/2303.07338) without pre-trained weights. 
The training process is as follows: train the model with cross-entropy in the first stage and replace the classifier with prototypes for all the classes in the subsequent stages. 
Please refer to the original implementation (https://github.com/zhoudw-zdw/RevisitingCIL) if you are using pre-trained weights.
'''
import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleCosineIncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy


num_workers = 8
batch_size = 32
milestones = [40, 80]

class SimpleCIL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleCosineIncrementalNet(args, False)
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self._total_classes = len(checkpoint["classes"])
        self.class_list = np.array(checkpoint["classes"])
        self.label_list = checkpoint["label_list"]
        print("Class list: ", self.class_list)
        self._network.update_fc(self._total_classes)
        self._network.load_checkpoint(checkpoint["network"])
        self._network.to(self._device)

    def after_task(self):
        self._known_classes = self._total_classes
    
    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "classes": self.data_manager.get_class_list(self._cur_task),
            "network": {
                "convnet": self._network.convnet.state_dict(),
                "fc": self._network.fc.state_dict()
            },
            "label_list": self.data_manager.get_label_list(self._cur_task),
        }
        torch.save(save_dict, "./{}/{}_{}.pkl".format(filename, self.args['model_name'], self._cur_task))
    
    def replace_fc(self,trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = model(data)["features"]
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = torch.nonzero(label_list == class_index).squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            if len(self._multiple_gpus) > 1:
              self._network.module.fc.weight.data[class_index] = proto
            else:
              self._network.fc.weight.data[class_index] = proto
        return model

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self.class_list = np.array(data_manager.get_class_list(self._cur_task))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"]
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args['init_epoch'], eta_min=self.min_lr
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        self.replace_fc(train_loader_for_protonet, self._network, None)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            elapsed = prog_bar.format_dict["elapsed"]
            rate = prog_bar.format_dict["rate"]
            remaining = (prog_bar.total - prog_bar.n) / rate if rate and prog_bar.total else 0  # Seconds*
            prog_bar.set_description(info)
            logging.info("Working on task {}: {:.2f}:{:.2f}".format(
                    self._cur_task,
                    elapsed,
                    remaining))            
        logging.info(info)
        logging.info("Finised on task {}: {:.2f}".format(
                    self._cur_task, elapsed))

   
