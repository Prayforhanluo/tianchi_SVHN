# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:55:20 2020

@author: LuoHan
"""


import os, sys, json
import numpy as np
import pandas as pd
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DataLoad import DataLoader
from model import *
from utils import *


if __name__ == '__main__':
    
    root = '../'
    num_workers = 2
    print ("===============build loader================")
    
    model = SVHN_model11()
    
    use_cuda = True
    train_loader = DataLoader(root, dtype = 'train', num_workers=num_workers)
    val_loader = DataLoader(root, dtype = 'val', num_workers=num_workers, )
    
    best_loss = 10000
    best_acc = 0
    # optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-4)
    
    for epoch in range(60):
        if epoch < 25:
            optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-4)
        elif epoch < 45:
            optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 1e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr = 0.00001, weight_decay = 1e-4)
        
        train_loss = Train(train_loader, model,optimizer, use_cuda = use_cuda)
        val_loss, val_predict_label, val_label= Validation(val_loader, model, 
                                                           use_cuda = use_cuda)
        val_label_pred = []
        for x in val_predict_label:
            tmp = ''
            for y in x:
                if y != 10:
                    tmp+=str(y)
                else:
                    break
            val_label_pred.append(tmp)
        val_label_true = []
        for x in val_label:
            val_label_true.append(''.join(map(str, x[x!=10])))
        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label_true))
    
        print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, 
                                        np.mean(train_loss), np.mean(val_loss)))
        print('Final Validation Acc : %s' % val_char_acc)
    
    torch.save(model.state_dict(), './SVHN_model11.pth')
