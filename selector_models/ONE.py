# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ONE(torch.nn.Module):
    def __init__(self, opt):
        super(ONE, self).__init__()

        self.opt = opt
        self.model_name = 'ONE'
        self.linear = torch.nn.Linear(self.opt.encoder_out_dimension, self.opt.rel_num)
        
    def forward(self, x):
        max_id = 0
        if (x.size(0) > 1):
            tmp = []
            for i in range(x.size(0)):
                # print(torch.max(self.linear(x[i]).squeeze(), 0)[0])
                tmp.append(torch.max(self.linear(x[i]).squeeze(), 0)[0])
            max_id = tmp.index(max(tmp))
            x = self.linear(x[max_id]).unsqueeze(0)
        else:
            x = self.linear(x)
        return x

    def select_max(self, x):
        max_id = 0
        return max_id