# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from models.resnet import ResNet


class Flow(nn.Module):
    def __init__(self):
        super(Flow, self).__init__()

    def forward(self, x, theta):
        cs = th.cos(theta * np.pi * 2)
        ss = th.sin(theta * np.pi * 2)
        return (1 + ss) * x + cs


class HyperbolicBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.theta_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * HyperbolicBasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * HyperbolicBasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != HyperbolicBasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * HyperbolicBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * HyperbolicBasicBlock.expansion)
            )

        self.flow = Flow()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.flow(self.shortcut(x), self.theta_function(x)))


class HyperbolicBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.theta_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * HyperbolicBottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * HyperbolicBottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * HyperbolicBottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * HyperbolicBottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * HyperbolicBottleNeck.expansion)
            )

        self.flow = Flow()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.flow(self.shortcut(x), self.theta_function(x)))


def hresnet18():
    """ return a ResNet 18 object
    """
    return ResNet(HyperbolicBasicBlock, [2, 2, 2, 2])

def hresnet34():
    """ return a ResNet 34 object
    """
    return ResNet(HyperbolicBasicBlock, [3, 4, 6, 3])

def hresnet50():
    """ return a ResNet 50 object
    """
    return ResNet(HyperbolicBottleNeck, [3, 4, 6, 3])

def hresnet101():
    """ return a ResNet 101 object
    """
    return ResNet(HyperbolicBottleNeck, [3, 4, 23, 3])

def hresnet152():
    """ return a ResNet 152 object
    """
    return ResNet(HyperbolicBottleNeck, [3, 8, 36, 3])
