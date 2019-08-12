# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ATT(torch.nn.Module):
    def __init__(self, opt):
        super(ATT, self).__init__()

        self.opt = opt
        self.model_name = 'ATT'
        self.query_r = torch.randn([opt.rel_num])   # According to the author's reply on github, they use only one query_r
        self.linear = torch.nn.Linear(self.opt.encoder_out_dimension, self.opt.rel_num)
        self.bilinear = torch.nn.Bilinear(self.opt.rel_num, self.opt.rel_num, 1)

    def forward(self, x):
        # print("x.size(): ", x.size())
        
        x = self.linear(x)
        if x.size(0) > 1:
            # print("x.size(): ", x.size())
            alpha = torch.empty(0)
            # print(alpha.size())
            if self.opt.use_gpu:
                alpha = alpha.cuda()
                self.query_r = self.query_r.cuda()
            for i in range(x.size(0)):
                # print("alpha: ", alpha)
                # print("Bilinear: ", self.bilinear(x[i], self.query_r))
                # alpha: [1, n] x: [n, 53]
                alpha = torch.cat((alpha, self.bilinear(x[i], self.query_r)), 0)
            # print("alpha.size(): ",alpha.size(), "alpha: ", alpha)
            alpha = F.softmax(alpha.unsqueeze(0), 1)
            # print(x.size(0), alpha.size(), x.size())
            x = torch.mm(alpha, x)
        # x = F.softmax(x, 1)
        # print("final x size: ", x.size())
        
        return x