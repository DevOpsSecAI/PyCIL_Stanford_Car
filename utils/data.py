import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels

import os

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )   
        
        
class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(25, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
        transforms.ColorJitter(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.470, 0.460, 0.455],
                std=[0.267, 0.266, 0.270]
            ),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class StanfordCar(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(320),
        transforms.CenterCrop(320),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(25, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
        transforms.ColorJitter(),
    ]
    test_trsf = [
        transforms.Resize(320),
        transforms.CenterCrop(320),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.470, 0.460, 0.455],
                std=[0.267, 0.266, 0.270]
            ),
    ]
    class_order = np.arange(196).tolist()
    def download_data(self):
        path = './car_data/car_data'
        train_dset = datasets.ImageFolder(os.path.join(path, "train"))
        test_dset = datasets.ImageFolder(os.path.join(path, "test"))
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class GeneralDataset(iData):
    def __init__(
        self,
        path,
        init_class_list = [-1],
        train_transform = None, 
        test_transform = None, 
        common_transform = None):
        self.use_path = True
        self.path = path
        self.train_trsf = train_transform
        if self.train_trsf == None:
            self.train_trsf = [
                transforms.RandomAffine(25, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness = 0.3, saturation = 0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))], p=1),  # Apply Gaussian blur with random probability
            ]
        self.test_trsf = test_transform
        if self.test_trsf == None:
            self.test_trsf = [
                transforms.Resize(224),
                transforms.CenterCrop(224),
            ]
        self.common_trsf = common_transform
        if self.common_trsf == None:
            self.common_trsf = [
                transforms.ToTensor(),
                transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    ),
            ]
        self.init_index = max(init_class_list) + 1
        self.class_order = np.arange(self.init_index, self.init_index + len(os.listdir(os.path.join(self.path, "train"))))
    
    def download_data(self):
        train_dset = datasets.ImageFolder(os.path.join(self.path, "train"))
        test_dset = datasets.ImageFolder(os.path.join(self.path, "val"))
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs, start_index = self.init_index)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs, start_index = self.init_index)
        return train_dset.classes

class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(320),
        transforms.CenterCrop(320),
    ]
    test_trsf = [
        transforms.Resize(320),
        transforms.CenterCrop(320),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
