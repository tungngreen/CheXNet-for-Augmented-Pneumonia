import os
import numpy as np
import time
import sys
import json
import collections

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import CheXNet
from config import config
from dataloader import dataloader
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from configparser import ConfigParser


#-------------------------------------------------------------------------------- 
class trainer:
    def __init__(self, config):
        super(trainer, self).__init__()
        self.config = config
        self.home_dir = os.getcwd()

        self.train_dir = config.train_dir
        self.test_dir = config.test_dir
        if config.augmented_dir != None:
            self.use_aug_data = True
            self.augmented_dir = config.augmented_dir
        else:
            self.use_aug_data = False
            self.augmented_dir = 'None'
        self.augmented_set =  config.augmented_set
        
        self.real_train_portion = config.real_train_portion
        self.real_test_portion = config.real_test_portion
        self.aug_train_portion = config.aug_train_portion

        self.use_cuda = config.use_cuda
        self.n_gpu = config.n_gpu
        self.gpu_id = config.gpu_id
        self.random_seed = config.random_seed
        self.is_traing = config.train
        self.is_testing = config.test
        self.test_after_training = config.test_after_training

        self.dense_net = config.dense_net
        self.is_trained = False
        self.n_class = config.n_class
        self.optimizer = config.optimizer

        self.image_size = config.image_size
        self.crop_size = config.crop_size
        
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.lr = config.lr
        self.parallel_training = config.parallel_training
        self.load_ckpt = True#config.load_ckpt
        self.ckpt_path = config.ckpt_path
        self.class_list = ['NORMAL', 'PNEUMONIA']

        if self.use_cuda:
            #torch.set_default_tensor_type('torch.cuda.FloatTensor')
            #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
            self.device = 'cuda:' + self.gpu_id
        else:
            #torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.device = 'cpu'
        
        self.model = CheXNet(self.n_class, self.dense_net, self.is_trained)
        if self.parallel_training & self.use_cuda:
            parallel_device_ids = []
            parallel_device_ids.append(int(self.gpu_id))
            self.model = torch.nn.DataParallel(self.model, device_ids = parallel_device_ids)
        
        
        

        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloader(self.train_dir, self.test_dir, self.use_aug_data, self.augmented_dir, self.real_train_portion, self.real_test_portion, self.aug_train_portion, self.batch_size, self.image_size, self.crop_size, self.class_list)
        self.save_dir = os.path.join(self.home_dir, 'save')
        
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if self.use_aug_data:
            self.save_dir = os.path.join(self.save_dir, self.augmented_set)

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        indices = [int(i) for i in os.listdir(self.save_dir)]
        if indices == []:
            save_index = str(1)
        else:    
            save_index = str(int(max(indices) + 1))
        self.save_index_dir = os.path.join(self.save_dir, save_index)
        os.mkdir(self.save_index_dir)
        config_file = os.path.join(self.save_index_dir, 'config.txt')
        with open(config_file, 'w') as f:
            #f.write('[CheXNet]\n')
            for arg, value in vars(config).items():
                f.write('--' + arg + '=' + str(value) + '\n')
                print('--' + arg + '=' + str(value))

    def train(self):
        device = self.device
        self.optimizer = Adam(self.model.parameters(), lr = self.lr, betas = (0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(self.optimizer, factor = 0.1, patience = 5, mode = 'min')
        loss_fn = torch.nn.BCELoss(reduction = 'mean')

        if self.load_ckpt:
            checkpoint = torch.load(self.ckpt_path, map_location = device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        save_dir = self.save_index_dir
        log_file = os.path.join(save_dir, 'log.txt')
        min_loss = 10000
        for epoch in range(0, self.n_epoch):
            start_time = time.time()
            loss_tensor_mean = 0
            loss_total = 0
            self.model.train()
            outGT = torch.FloatTensor().to(device)
            outPred = torch.FloatTensor().to(device)
            for batch_id, (x, y) in enumerate(self.train_dataloader):
                x = x.to(device)
                y = y.to(device)

                pred = self.model(x)
                loss = loss_fn(pred, y)

                loss_tensor_mean += loss
                loss_total += loss.data.item()

                outGT = torch.cat((outGT, y), 0).to(device)
                outPred = torch.cat((outPred, pred), 0).to(device)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_train_mean = loss_total / (batch_id + 1)
            #loss_tensor_mean = loss_tensor_mean / (batch_id + 1)

            pred = torch.max(outPred, 1)[1].cpu()
            truth = torch.max(outGT, 1)[1].cpu()
            train_acc = accuracy_score(truth, pred)
            train_cm = confusion_matrix(truth, pred)


            loss_total = 0
            loss_tensor = 0
            self.model.eval()
            outGT = torch.FloatTensor().to(device)
            outPred = torch.FloatTensor().to(device)
            with torch.no_grad():
                for batch_id, (x, y) in enumerate(self.val_dataloader):
                    x = x.to(device)
                    y = y.to(device)

                    pred = self.model(x)
                    loss = loss_fn(pred, y)
                    loss_total += loss.data.item()
                    loss_tensor += loss

                    outGT = torch.cat((outGT, y), 0).to(device)
                    outPred = torch.cat((outPred, pred), 0).to(device)

            loss_val_mean = loss_total / (batch_id + 1)
            loss_tensor_mean = loss_tensor / (batch_id + 1)

            pred = torch.max(outPred, 1)[1].cpu()
            truth = torch.max(outGT, 1)[1].cpu()
            val_acc = accuracy_score(truth, pred)
            val_cm = confusion_matrix(truth, pred)

            scheduler.step(loss_tensor_mean.data.item())

            curr_time = time.time()
            with open(log_file, 'a') as f:
                print('Epoch [' + str(epoch + 1) + '] [train] loss = ' + str(loss_train_mean) + '| [time] ' + str(curr_time - start_time) + ' | [accu] ' + str(train_acc) + ' | [matrix] {} {} {} {}'.format(train_cm[0, 0], train_cm[0, 1], train_cm[1, 0], train_cm[1, 1]))
                f.write(('Epoch [' + str(epoch + 1) + '] [train] loss = ' + str(loss_train_mean) + '| [time] ' + str(curr_time - start_time) + ' | [accu] ' + str(train_acc) + ' | [matrix] {} {} {} {}\n'.format(train_cm[0, 0], train_cm[0, 1], train_cm[1, 0], train_cm[1, 1])))

                if loss_val_mean < min_loss:
                    min_loss = loss_val_mean
                    torch.save({'epoch': epoch + 1, 'state_dict': self.model.state_dict(), 'best_loss': min_loss, 'optimizer' : self.optimizer.state_dict()}, save_dir + '/' + 'm-' + 'best' + '.pth.tar')
                    print(('Epoch [' + str(epoch + 1) + '] [best] loss = ' + str(loss_val_mean) + '| [time] ' + str(curr_time - start_time) + ' | [accu] ' + str(val_acc) + ' | [matrix] {} {} {} {}'.format(val_cm[0, 0], val_cm[0, 1], val_cm[1, 0], val_cm[1, 1])))
                    f.write(('Epoch [' + str(epoch + 1) + '] [best] loss = ' + str(loss_val_mean) + '| [time] ' + str(curr_time - start_time) + ' | [accu] ' + str(val_acc) + ' | [matrix] {} {} {} {}\n'.format(val_cm[0, 0], val_cm[0, 1], val_cm[1, 0], val_cm[1, 1])))

                else:
                    print(('Epoch [' + str(epoch + 1) + '] [-----] loss = ' + str(loss_val_mean) + '| [time] ' + str(curr_time - start_time) + ' | [accu] ' + str(val_acc) + ' | [matrix] {} {} {} {}'.format(val_cm[0, 0], val_cm[0, 1], val_cm[1, 0], val_cm[1, 1])))
                    f.write(('Epoch [' + str(epoch + 1) + '] [-----] loss = ' + str(loss_val_mean) + '| [time] ' + str(curr_time - start_time) + ' | [accu] ' + str(val_acc) + ' | [matrix] {} {} {} {}\n'.format(val_cm[0, 0], val_cm[0, 1], val_cm[1, 0], val_cm[1, 1])))

                if epoch % 15 == 0:
                    torch.save({'epoch': epoch + 1, 'state_dict': self.model.state_dict(), 'best_loss': min_loss, 'optimizer' : self.optimizer.state_dict()}, save_dir + '/' + 'epoch' + str(epoch) + '.pth.tar')
                
            f.close()


    def test(self):

        device = self.device
        outGT = torch.FloatTensor().to(device)
        outPred = torch.FloatTensor().to(device)

        if self.test_after_training:
            ckpt_path = os.path.join(self.save_index_dir, 'm-best.pth.tar')
            ckpt = torch.load(ckpt_path, map_location = self.device)
            self.model.load_state_dict(ckpt['state_dict'])

        self.model.eval()
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(self.test_dataloader):
                x = x.to(device)
                y = y.to(device)
                bs, n_crops, c, h, w = x.size()
                input = x.view(-1, c, h, w).to(device)
                pred = self.model(input)

                outGT = torch.cat((outGT, y), 0).to(device)
                predMean = pred.view(bs, n_crops, -1).mean(1)
                outPred = torch.cat((outPred, predMean), 0).to(device)

        pred = torch.max(outPred, 1)[1].cpu()
        truth = torch.max(outGT, 1)[1].cpu()

        result_log_file = os.path.join(self.save_index_dir, 'result_log_file.txt')
        with open(result_log_file, 'w') as f:
            f.write('accu = {}\n'.format(str(accuracy_score(truth, pred))))
            cm = confusion_matrix(truth, pred)
            f.write('Confusion_matrix (TN, FP, FN, TP): {} {} {} {}\n'.format(cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]))
            f.close()


    
    


if __name__ == '__main__':
    #print(config)
    trainer = trainer(config)
    trainer.train()
    trainer.test()
    # temp = 0
    # for x, y in trainer.train_dataloader:
    #     temp = temp + len(x)
    # print(temp)





