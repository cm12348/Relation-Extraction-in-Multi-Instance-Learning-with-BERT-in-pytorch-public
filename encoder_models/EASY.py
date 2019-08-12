# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EASY(torch.nn.Module):
    def __init__(self, opt):
        super(EASY, self).__init__()

        self.opt = opt
        self.model_name = 'EASY'
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.embed_dim)
        self.pos1_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        self.pos2_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)

        feature_dim = self.opt.embed_dim + self.opt.pos_dim * 2

        n_class = self.opt.rel_num
        kernels=[2,3,4]
        kernel_number=[30,30,30]
        self.convs = nn.ModuleList([nn.Conv2d(1, number, (size, feature_dim),padding=(size-1,0)) for (size,number) in zip(kernels,kernel_number)])
        self.dropout=nn.Dropout()
        self.out_dimension = sum(kernel_number)
        self.out = nn.Linear(self.out_dimension, n_class)
        
        self.init_model_weight()
        self.init_word_emb()

    def forward(self, insX, insPFs):
        insPF1, insPF2 = torch.split(insPFs, 1)

        word_emb = self.word_embs(insX)
        pf1_emb = self.pos1_embs(insPF1)
        pf2_emb = self.pos2_embs(insPF2)

        x = torch.cat([word_emb, pf1_emb.squeeze(), pf2_emb.squeeze()], 1)
        x = x.unsqueeze(0).unsqueeze(0)
        # print("x.size(): ", x.size())

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # print("x.size(): ", x.size())
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # print("x.size(): ", x.size())
        x = torch.cat(x, 1)
        # print("x.size(): ", x.size())
        x = self.dropout(x)
        # print("x.size(): ", x.size())
        # x = self.out(x)
        # print("x.size(): ", x.size())

        return x
    def init_model_weight(self):
        '''
        use xavier to init
        '''
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)

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