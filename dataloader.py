import os
import torch
import time
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms

class loader(Dataset):
    def __init__(self, dir_path, img_size, crop_size, class_list, stage):
        self.img_size = img_size
        self.crop_size = crop_size
        self.class_list = class_list
        self.clas = dir_path.split('/')[-1]
        self.stage = stage

        self.image_list = []
        self.label_list = []

        label_sample_array = np.zeros((len(class_list), ), dtype = int)
        for file in os.listdir(dir_path):
            img_path = os.path.join(dir_path, file)
            label = label_sample_array
            label[self.class_list.index(self.clas)] = 1

            self.image_list.append(img_path)
            self.label_list.append(label)

    def transform(self, img):
        if self.stage == 'train':
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.TenCrop(self.crop_size),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))
                ]
            )
        return transform(img)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = self.transform(Image.open(image_path).convert('RGB'))
        label = torch.FloatTensor(self.label_list[index])
        return image, label

    def __len__(self):
        return len(self.image_list)

def get_portion(data_dir, portion_to_get, img_size, crop_size, class_list, stage):
    dataset = loader(data_dir, img_size, crop_size, class_list, stage)
    dset_len = len(dataset)
    split_len = int(dset_len * portion_to_get)
    dropped_len = dset_len - split_len
    #print(dset_len, split_len, dropped_len)
    lengths = [split_len, dropped_len]

    split_dataset, _ = random_split(dataset = dataset, lengths = lengths)
    return split_dataset

def dataloader(train_dir, test_dir, use_aug_data, augmented_dir, real_train_portion, real_test_portion, aug_train_portion, batch_size, img_size, crop_size, class_list):
    train_dsets = []
    test_dsets = []
    dir_list = [train_dir, test_dir, augmented_dir]
    portion_list = [real_train_portion, real_test_portion, aug_train_portion]

    for portion, dset_dir in zip(portion_list, dir_list):
        for clas in class_list:
            class_dir = os.path.join(dset_dir, clas)
            if os.path.isdir(class_dir) == False:
                continue
            
            if clas != 'NORMAL':
                portion = 1
            if 'test' in dset_dir:
                stage = 'test'
            else:
                stage = 'train'
            class_dset = get_portion(class_dir, portion, img_size, crop_size, class_list, stage)
        
            if 'test' in dset_dir:
                test_dsets.append(class_dset)
            else:
                train_dsets.append(class_dset)
    train_dsets = ConcatDataset(train_dsets)

    train_dset_len = len(train_dsets)
    train_len = int(train_dset_len * 0.8)
    val_len = train_dset_len - train_len
    lengths = [train_len, val_len]
    train_dsets, val_dsets = random_split(dataset = train_dsets, lengths = lengths)
    train_dloader = DataLoader(dataset = train_dsets, batch_size = batch_size, shuffle = True, pin_memory = True)

    val_dloader = DataLoader(dataset = val_dsets, batch_size = batch_size, shuffle = False, pin_memory = True)

    test_dsets = ConcatDataset(test_dsets)
    test_dloader = DataLoader(dataset = test_dsets, batch_size = batch_size, shuffle = False, pin_memory = True)
  
    return train_dloader, val_dloader, test_dloader





        
        

    

    

