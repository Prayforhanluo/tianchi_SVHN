# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:13:17 2020

@author: LuoHan
"""


import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
from ResNetCBAM import *
import torch.nn as nn
import torch.nn.functional as F
import math


 
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ECABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(planes * 4, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, k_size=[3, 3, 3, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def eca_resnet18(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(ECABasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet34(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(ECABasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet50(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Constructing eca_resnet50......")
    model = ResNet(ECABottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet101(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(ECABottleneck, [3, 4, 23, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet152(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(ECABottleneck, [3, 8, 36, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model





class BottleneckResidualSEBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):

        shortcut = self.shortcut(x)

        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)


class SEResNet(nn.Module):

    def __init__(self, block, block_num, class_num=100):
        super().__init__()

        self.in_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 516, 2)

        self.linear = nn.Linear(self.in_channels, class_num)
    
    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        # x = self.linear(x)

        return x

    
    def _make_stage(self, block, num, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1))
            num -= 1
        
        return nn.Sequential(*layers)


def SeResNet50():
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 6, 3])





##############################################################################
###                             SVHN Models                                ###
##############################################################################

class SVHN_model1(nn.Module):
    """
        采用 baseline的方式用预训练的res18作为model.
        score : 0.67
    """
    def __init__(self):
        super(SVHN_model1, self).__init__()
        model_conv = models.resnet18(pretrained = True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1]) # 去掉最后一层全连接
        
        self.cnn = model_conv
        
        self.fc1 = nn.Sequential(nn.Linear(512, 256),
                                  nn.ReLU(),
                                  nn.Linear(256,11)) #第一个数不可能为 *
        self.fc2 = nn.Sequential(nn.Linear(512, 256),
                                  nn.ReLU(),
                                  nn.Linear(256,11))
        self.fc3 = nn.Sequential(nn.Linear(512, 256),
                                  nn.ReLU(),
                                  nn.Linear(256,11))
        self.fc4 = nn.Sequential(nn.Linear(512, 256),
                                  nn.ReLU(),
                                  nn.Linear(256,11))
        
        # self.fc1 = nn.Linear(512,10)
        # self.fc2 = nn.Linear(512,11)
        # self.fc3 = nn.Linear(512,11)
        # self.fc4 = nn.Linear(512,11)
        # self.fc5 = nn.Linear(512,11)
    
    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        pred = torch.stack((c1,c2,c3,c4), dim=1)
        
        
        return pred


class SVHN_model2(nn.Module):
    """
        采用预训练的 res50 net来提取特征
    """
    def __init__(self):
        super(SVHN_model2, self).__init__()
        model_conv = models.resnet50(pretrained = True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1]) # 去掉最后一层全连接
        
        self.cnn = model_conv
        
        self.fc1 = nn.Sequential(nn.Linear(2048, 256),
                                  nn.ReLU(),
                                  nn.Linear(256,128),
                                  nn.Dropout(0.5),
                                  nn.ReLU(),
                                  nn.Linear(128,11))
        
        self.fc2 = nn.Sequential(nn.Linear(2048, 256),
                                  nn.ReLU(),
                                  nn.Linear(256,128),
                                  nn.Dropout(0.5),
                                  nn.ReLU(),
                                  nn.Linear(128,11))
        
        self.fc3 = nn.Sequential(nn.Linear(2048, 256),
                                  nn.ReLU(),
                                  nn.Linear(256,128),
                                  nn.Dropout(0.5),
                                  nn.ReLU(),
                                  nn.Linear(128,11))
        
        self.fc4 = nn.Sequential(nn.Linear(2048, 256),
                                  nn.ReLU(),
                                  nn.Linear(256,128),
                                  nn.Dropout(0.5),
                                  nn.ReLU(),
                                  nn.Linear(128,11))
        
        # self.fc1 = nn.Linear(512,10)
        # self.fc2 = nn.Linear(512,11)
        # self.fc3 = nn.Linear(512,11)
        # self.fc4 = nn.Linear(512,11)
        # self.fc5 = nn.Linear(512,11)
    
    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        pred = torch.stack((c1,c2,c3,c4), dim=1)
        
        return pred 


class SVHN_model3(nn.Module):
    """
        采用预训练的 dense121 net来提取特征
    """
    def __init__(self):
        super(SVHN_model3, self).__init__()
        model_conv = models.densenet121(pretrained = True)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        
        self.cnn = model_conv
        
        self.fc1 = nn.Linear(3072,11)
        self.fc2 = nn.Linear(3072,11)
        self.fc3 = nn.Linear(3072,11)
        self.fc4 = nn.Linear(3072,11)
        
    
    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        pred = torch.stack((c1,c2,c3,c4), dim=1)

        return pred
    
class SVHN_model4(nn.Module):
    """
        efficient net not pre-trained
    """
    def __init__(self):
        super(SVHN_model4, self).__init__()
        model_conv = EfficientNet.from_pretrained('efficientnet-b7')
        self.cnn = model_conv
        
        
        # self.fc0 = nn.Linear(1000, 4)
        self.fc1 = nn.Linear(1000,11)
        self.fc2 = nn.Linear(1000,11)
        self.fc3 = nn.Linear(1000,11)
        self.fc4 = nn.Linear(1000,11)
        # self.fc5 = nn.Linear(1000,11)
        
    def forward(self, x):
        feat = self.cnn(x)
        
        # L = self.fc0(feat)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        pred = torch.stack((c1,c2,c3,c4), dim=1)

        return pred


class SVHN_model5(nn.Module):
    """
         ECA resnet 50
    """
    def __init__(self):
        super(SVHN_model5, self).__init__()
        model_conv = eca_resnet50()
        self.cnn = model_conv
        
        self.fc1 = nn.Linear(1000,11)
        self.fc2 = nn.Linear(1000,11)
        self.fc3 = nn.Linear(1000,11)
        self.fc4 = nn.Linear(1000,11)
    
    def forward(self, x):
        feat = self.cnn(x)
        
        c1 = self.fc1(feat)   
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        
        pred = torch.stack((c1,c2,c3,c4), dim=1)

        return pred

class SVHN_model6(nn.Module):
    """
    """
    def __init__(self):
        super(SVHN_model6, self).__init__()
        cnn = cbam_resnet101()
        cnn_modules = list(cnn.children())[:-1]
        cnn = nn.Sequential(*cnn_modules)
        self.cnn = cnn
        
        # Linear 进行shape 变化
        self.fc1 = nn.Linear(2048,11)
        self.fc2 = nn.Linear(2048,11)
        self.fc3 = nn.Linear(2048,11)
        self.fc4 = nn.Linear(2048,11)
        
    def forward(self, x):
        
        feat = self.cnn(x)
        feat = feat.view(feat.shape[0], -1)        
        c1 = self.fc1(feat)   
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        
        pred = torch.stack((c1,c2,c3,c4), dim=1)

        return pred

class SVHN_model7(nn.Module):
    """
    """
    def __init__(self):
        super(SVHN_model7, self).__init__()
        cnn = models.resnet18(pretrained = True)
        cnn_modules = list(cnn.children())[:-1]
        cnn = nn.Sequential(*cnn_modules)
        self.cnn = cnn
        
        # Linear 进行shape 变化
        self.fc1 = nn.Sequential(nn.Linear(512,64),
                                 nn.BatchNorm1d(64, momentum=0.01))
        
        self.lstm = nn.LSTM(64,256,2, bidirectional = True)
        self.fc2 = nn.Linear(512, 11)
        
    def forward(self, x):
        
        x = self.cnn(x) #卷积取特征
        x = x.view(x.shape[0], -1) #二维展开
        
        x = self.fc1(x) # 过Linear变换至LSTM的shape
        x = x.expand(4,x.shape[0],x.shape[1]) # 5个数字的序列 expand
        
        x, (hn, cn) = self.lstm(x) #LSTM 层
        x = self.fc2(x)         #全连接
        x = x.permute([1,0,2])
    
        return x 

class SVHN_model8(nn.Module):
    """
    """
    def __init__(self):
        super(SVHN_model8, self).__init__()
        cnn = eca_resnet50()
        self.cnn = cnn
        
        # Linear 进行shape 变化
        self.fc1 = nn.Sequential(nn.Linear(1000,64),
                                 nn.BatchNorm1d(64, momentum=0.01))
        
        self.lstm = nn.LSTM(64,256,2, bidirectional = True)
        self.fc2 = nn.Linear(512, 11)
        
    def forward(self, x):
        
        x = self.cnn(x) #卷积取特征
        x = x.view(x.shape[0], -1) #二维展开
        
        x = self.fc1(x) # 过Linear变换至LSTM的shape
        x = x.expand(4,x.shape[0],x.shape[1]) # 5个数字的序列 expand
        
        x, (hn, cn) = self.lstm(x) #LSTM 层
        x = self.fc2(x)         #全连接
        x = x.permute([1,0,2])
    
        return x


class SVHN_model9(nn.Module):
    """
    """
    def __init__(self):
        super(SVHN_model9, self).__init__()
        self.cnn = SeResNet50()
        
        # Linear 进行shape 变化
        self.fc1 = nn.Linear(2064,11)
        self.fc2 = nn.Linear(2064,11)
        self.fc3 = nn.Linear(2064,11)
        self.fc4 = nn.Linear(2064,11)
        
    def forward(self, x):
        
        feat = self.cnn(x)
        # feat = feat.view(feat.shape[0], -1)        
        c1 = self.fc1(feat)   
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        
        pred = torch.stack((c1,c2,c3,c4), dim=1)

        return pred
    
class SVHN_model10(nn.Module):
    """
    """
    def __init__(self):
        super(SVHN_model10, self).__init__()
        self.cnn = cbam_resnet18()
        self.relu = nn.ReLU()
        # Linear 进行shape 变化
        self.fc1 = nn.Linear(1000,11)
        self.fc2 = nn.Linear(1000,11)
        self.fc3 = nn.Linear(1000,11)
        self.fc4 = nn.Linear(1000,11)
    
    def forward(self, x):
        feat = self.cnn(x)
        feat = self.relu(feat)
        
        c1 = self.fc1(feat)   
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        
        pred = torch.stack((c1,c2,c3,c4), dim=1)
        
        return pred
    

class SVHN_model11(nn.Module):
    """
    """
    def __init__(self):
        super(SVHN_model11, self).__init__()
        cnn = cbam_resnet101()
        
        ## CNN part
        self.conv1 = cnn.conv1
        self.bn1 = cnn.bn1
        self.relu = cnn.relu
        self.maxpool = cnn.maxpool
        self.ca = cnn.ca
        self.sa = cnn.sa
        self.ca1 = cnn.ca1
        self.sa1 = cnn.sa1
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4
        
        ## RNN part
        self.rnn = nn.GRU(2048, 256, num_layers=2, 
                          batch_first = False,
                          bidirectional = True)
        
        self.fc = nn.Linear(512, 11)
    
    def forward(self, x):
        #第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 第一层 attention
        x = self.ca(x) * x
        x = self.sa(x) *x
        # layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #最后一层 attention
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        
        #RNN 层
        x = x.mean(2)
        x = x.permute(2,0,1)
        x, hidden = self.rnn(x)
        
        x = self.fc(x)
        x = x.permute(1,0,2)
        
        return x
    
        
    
    
        