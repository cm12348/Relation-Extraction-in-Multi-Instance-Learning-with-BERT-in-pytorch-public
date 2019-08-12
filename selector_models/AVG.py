# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AVG(torch.nn.Module):
    def __init__(self, opt):
        super(AVG, self).__init__()

        self.opt = opt
        self.model_name = 'AVG'
        self.linear = torch.nn.Linear(self.opt.encoder_out_dimension, self.opt.rel_num)

    def forward(self, x):
        if (x.size(0) > 1):
            x = torch.mean(x, 0, True)
        # print(x.size())
        x = self.linear(x)
        # print(x.size())
        return x

        