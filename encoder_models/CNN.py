# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(torch.nn.Module):
    def __init__(self, opt):
        super(CNN, self).__init__()

        self.opt = opt
        self.model_name = 'CNN'
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.embed_dim)
        self.pos1_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        self.pos2_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        
        feature_dim = self.opt.embed_dim + self.opt.pos_dim * 2
        # print(feature_dim)

        self.c1 = nn.Conv2d(1, self.opt.num_filters, (3, feature_dim), padding=2)
        self.mp1 = nn.MaxPool2d(3,3)
        self.mp2 = nn.MaxPool1d(27)
        self.c2 = nn.Sequential(
                nn.Conv2d(6, 16, (3, self.opt.embed_dim), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3)
            )

        self.linear = nn.Linear(self.opt.num_filters, self.opt.rel_num)
        self.dropout = nn.Dropout(self.opt.drop_out)

        # self.init_model_weight()
        self.init_word_emb()

    def forward(self, insX, insPFs):
        # insX, insPFs = x
        # print("insX.size(): ", insX.size())
        # print("indPFs.size(): ", insPFs.size())
        insPF1, insPF2 = torch.split(insPFs, 1)

        word_emb = self.word_embs(insX)
        pf1_emb = self.pos1_embs(insPF1)
        pf2_emb = self.pos2_embs(insPF2)

        x = torch.cat([word_emb, pf1_emb.squeeze(), pf2_emb.squeeze()], 1)
        # print("x.size(): ", x.size())

        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        # print("x.size(): ", x.size())
        
        x = self.c1(x)
        x = F.relu(x)
        x = x.squeeze(0)
        
        # print("x.size(): ", x.size())

        x = self.mp1(x)
        # print("x.size(): ", x.size())
        x = x.squeeze(2)
        x = x.unsqueeze(0)
        x = self.mp2(x)
        # print("x.size(): ", x.size())
        x = x.squeeze()
        x = self.dropout(x)
        x = self.linear(x)
        x = F.softmax(x)
        x = x.unsqueeze(0)
        return x

    def init_model_weight(self):
        '''
        use xavier to init
        '''
        nn.init.xavier_uniform_(self.c1.weight)

    def init_word_emb(self):

        def p_2norm(path):
            v = torch.from_numpy(np.load(path))
            if self.opt.norm_emb:
                v = torch.div(v, v.norm(2, 1).unsqueeze(1))
                v[v != v] = 0.0
            return v

        w2v = p_2norm(self.opt.w2v_path)
        p1_2v = p_2norm(self.opt.p1_2v_path)
        p2_2v = p_2norm(self.opt.p2_2v_path)

        if self.opt.use_gpu:
            self.word_embs.weight.data.copy_(w2v.cuda())
            self.pos1_embs.weight.data.copy_(p1_2v.cuda())
            self.pos2_embs.weight.data.copy_(p2_2v.cuda())
        else:
            self.pos1_embs.weight.data.copy_(p1_2v)
            self.pos2_embs.weight.data.copy_(p2_2v)
            self.word_embs.weight.data.copy_(w2v)
