# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:50:16 2020

@author: LuoHan
"""


import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


##########################################################

#  卷积层提特征， LSTM或fc 进行序列识别， 只计算字符+终止符的loss  #

##########################################################

def maskloss(pred, label, average = True, use_cuda = False):
    """
        只计算字符+终止符的loss
    """
    loss = nn.CrossEntropyLoss(reduction = 'none')
    stop_index = torch.Tensor(list(map(lambda x : len(x[x!=10]), label))).long() + 1
    label_mask = torch.randint(low=0,high=1, size=label.shape)
    for i in range(len(stop_index)):
        label_mask[i,0:stop_index[i]] = 1
    if use_cuda:
        label_mask = label_mask.cuda()
    loss1 = loss(pred[:,0,:], label[:,0].long())
    loss2 = loss(pred[:,1,:], label[:,1].long())
    loss3 = loss(pred[:,2,:], label[:,2].long())
    loss4 = loss(pred[:,3,:], label[:,3].long())
    loss_final = torch.stack((loss1, loss2, loss3, loss4))
    loss_final = loss_final.T * label_mask
    if average:
        loss_final = loss_final[loss_final != 0].mean()
    else:
        loss_final = loss_final[loss_final != 0].sum()
    
    return loss_final

class MaskCELoss(nn.Module):
    def __init__(self, average = True, use_cuda = False):
        super().__init__()
        self.average = average
        self.use_cuda = use_cuda
    def forward(self, pred, label):
        return maskloss(pred, label, average = self.average, use_cuda = self.use_cuda)
    


def Train(train_loader, model, optimizer, use_cuda = False):
    """
    """
    train_loss = []
    criterion = MaskCELoss(use_cuda = use_cuda)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    model.train()
    for batch, (img, label) in tqdm(enumerate(train_loader)):
        if use_cuda:
            img = img.cuda()
            label = label.cuda()
        
        pred = model(img)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        
    return train_loss 



def Validation(validation_loader, model, use_cuda = False):
    """
    """
    predict = []
    true_label= []
    validation_loss = []
    
    criterion = MaskCELoss(use_cuda = use_cuda)    
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    model.eval()
    
    with torch.no_grad():
        for batch, (img, label) in tqdm(enumerate(validation_loader)):
            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            
            pred = model(img)
            loss = criterion(pred, label)
            if use_cuda:
                predict.append(pred.argmax(dim=2).data.cpu().numpy())
                label = label.cpu().numpy()
            else:
                predict.append(pred.argmax(dim=2).data.numpy())
                label = label.numpy()

            validation_loss.append(loss.item())
            true_label.append(label)
        predict = np.vstack(predict)
        true_label = np.vstack(true_label)
            
    return validation_loss, predict, true_label


def Predication(prediction_loader, model, use_cuda = False):
    """
    """
    predict = []
    pred_pred = []
    if use_cuda:
        model = model.cuda()
    model.eval()
    
    with torch.no_grad():
        for batch, img in tqdm(enumerate(prediction_loader)):
            if use_cuda:
                img = img.cuda()
            
            pred = model(img)
            if use_cuda:
                predict.append(pred.argmax(dim=2).data.cpu().numpy())
                pred_pred.append(pred.data.cpu().numpy())
            else:
                predict.append(pred.argmax(dim=2).data.numpy())
                pred_pred.append(pred.data.numpy())

        predict = np.vstack(predict)
        pred_pred = np.vstack(pred_pred)
    return  predict, pred_pred


def Train2(train_loader, model, optimizer, use_cuda = False):
    """
    """
    train_loss = []
    # criterion = MaskCELoss(use_cuda = use_cuda)
    criterion = nn.CTCLoss(zero_infinity = True)
    sft = nn.LogSoftmax(dim=2)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    model.train()
    for batch, (img, label) in tqdm(enumerate(train_loader)):
        if use_cuda:
            img = img.cuda()
            label = label.cuda()
        
        pred = model(img)
        seq = label[label != 10]
        input_len =  torch.full(size=(40,), fill_value=4, dtype=torch.long)
        target_len = (label != 10).sum(axis=1)
        # loss = criterion(pred, label)
        loss = criterion(sft(pred).permute((1,0,2)), seq, input_len, target_len)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 10.0)
        optimizer.step()
        train_loss.append(loss.item())
        
    return train_loss 



def Validation2(validation_loader, model, use_cuda = False):
    """
    """
    predict = []
    true_label= []
    validation_loss = []
    
    # criterion = MaskCELoss(use_cuda = use_cuda)
    criterion = nn.CTCLoss(zero_infinity = True)
    sft = nn.LogSoftmax(dim=2)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    model.eval()
    
    with torch.no_grad():
        for batch, (img, label) in tqdm(enumerate(validation_loader)):
            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            pred = model(img)
            seq = label[label!=10]
            input_len =  torch.full(size=(40,), fill_value=4, dtype=torch.long)
            target_len = (label != 10).sum(axis=1)
            # loss = criterion(pred, label)
            loss = criterion(sft(pred).permute((1,0,2)), seq, input_len, target_len)
            if use_cuda:
                predict.append(pred.argmax(dim=2).data.cpu().numpy())
                label = label.cpu().numpy()
            else:
                predict.append(pred.argmax(dim=2).data.numpy())
                label = label.numpy()

            validation_loss.append(loss.item())
            true_label.append(label)
        predict = np.vstack(predict)
        true_label = np.vstack(true_label)
            
    return validation_loss, predict, true_label
