# -*- coding: utf-8 -*-

import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.modeling_bert import (BertForSequenceClassification,
                                                BertModel)
from sklearn import metrics
from torch.utils.data import DataLoader

import dataset
import encoder_models
import selector_models
from config import opt

# from pytorch_pretrained_bert.modeling import (BertForSequenceClassification,
#                                               BertModel)

warnings.filterwarnings(opt.filterwarning)

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

def train(**kwargs):
    # kwargs.update({'model': 'CNN'})
    opt.parse(kwargs)

    if (opt.use_gpu):
        torch.cuda.set_device(opt.gpu_id)

    if opt.encoder=='BERT':
        encoder_model = BertForSequenceClassification.from_pretrained("./downloaded_weights/downloaded_bert_base_uncased", num_labels=opt.rel_num)
        # print(encoder_model)
        opt.encoder_out_dimension = opt.rel_num
    else:
        encoder_model = getattr(encoder_models, opt.encoder)(opt)
        opt.encoder_out_dimension = encoder_model.out_dimension
    selector_model = getattr(selector_models, opt.selector)(opt)
    # encoder_model = torch.nn.DataParallel(encoder_model, device_ids=[3,6])

    if (opt.use_gpu):
        encoder_model = encoder_model.cuda()
        selector_model = selector_model.cuda()

    # Loading data
    DataModel = getattr(dataset, opt.data + 'Data')
    train_data = DataModel(opt.data_root, train=True, use_bert=opt.use_bert_tokenizer)
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    print('train data: {}'.format(len(train_data)))

    test_data = DataModel(opt.data_root, train=False, use_bert=opt.use_bert_tokenizer)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    print('test data: {}'.format(len(test_data)))
    
    
    criterion = nn.CrossEntropyLoss()
    if opt.encoder == 'BERT':
        optimizer = AdamW([
            {'params': encoder_model.parameters()},
            {'params': selector_model.parameters()}
            ], lr=opt.lr, correct_bias=True)  # To reproduce BertAdam specific behavior set correct_bias=False
    else:
        optimizer = optim.Adadelta([
            {'params': encoder_model.parameters()},
            {'params': selector_model.parameters()}
            ], lr=opt.lr, rho=1.0, eps=1e-6, weight_decay=opt.weight_decay)


    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=2, t_total=3)  # PyTorch scheduler
    ### and used like this:
    # for batch in train_data:
    #     loss = model(batch)
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
        
    #     optimizer.zero_grad()


    # if opt.encoder == "BERT" and False:
    #     optimizer = optim.SGD([
    #         {'params': selector_model.parameters()}
    #         ], lr=opt.lr)
    # else:
    
    optimizer = optim.SGD([
        {'params': encoder_model.parameters()},
        {'params': selector_model.parameters()}
        ], lr=opt.lr)

    max_pre = 0.0
    max_rec = 0.0
    for epoch in range(opt.num_epochs):
        # if opt.encoder == "BERT":
        encoder_model.train()
        selector_model.train()
        print("*"*50)
        print("Epoch {}".format(epoch))
        total_loss = 0
        max_insNum = 0
        for batch_num, (data, label_set) in enumerate(train_data_loader):
            # if (batch_num>2000):
            #     break
            # label_set is the label of each bag (there may be no more than 4 labels, but we only wants the first)

            labels = []
            outs = torch.empty([0, 53])

            empty = True # if all labels of bags in one batch are zeros, then it's empty, continue to avoid error
            for l in label_set:
                if (l[0] != 0):
                    labels.append(l[0])
                    empty = False
            if empty:
                continue
            # labels = [l[0] for l in label_set]
            # Each time enters {batch_size} bags
            # Each time I want one bag!!
            # The model need to give me a representation of an instance!!!

            if opt.use_gpu:
                labels = torch.LongTensor(labels).cuda()
                outs = outs.cuda()
            else:
                labels = torch.LongTensor(labels)

            
            optimizer.zero_grad()
            train_cor = 0
            for idx, bag in enumerate(data):
                insNum = bag[1]
                # if insNum > max_insNum:
                #     max_insNum = insNum
                #     print(max_insNum)
                label = label_set[idx][0] # Label of the current bag  
                if (label_set[idx][0] == 0):
                    continue                      
                
                ins_outs = torch.empty(0)
                instances = bag[2]
                pf_list = []
                mask_list = []
                if opt.encoder!='BERT':
                    pf_list = bag[3]
                    mask_list = bag[5]

                # pf_list = bag[3]
                ins_out = torch.empty(0)
                encoder_model.batch_size = insNum
                if opt.use_gpu:
                    instances = torch.LongTensor(instances).cuda()

                if opt.encoder == 'BERT':
                    # with torch.no_grad():
                        # print(instances.size(0))
                    if insNum > opt.max_sentence_in_bag:
                        ins_outs = encoder_model(instances[:opt.max_sentence_in_bag])[0]
                    else:
                        ins_outs = encoder_model(instances)[0]
                    # ins_outs = ins_outs[0]
                    # print(ins_outs[0].size())
                else:

                    for idx, instance in enumerate(instances):
                        if opt.use_gpu:
                            pfs = torch.LongTensor(pf_list[idx]).cuda()
                            masks = torch.LongTensor(mask_list[idx]).cuda()
                        else:
                            pfs = torch.LongTensor(pf_list[idx])
                            masks = torch.LongTensor(mask_list[idx])

                        if opt.encoder=='PCNN':
                            ins_out = encoder_model(instance, pfs, masks)
                        else:
                            ins_out = encoder_model(instance, pfs)

                        if (opt.use_gpu):
                            ins_out = ins_out.cuda()
                            ins_outs = ins_outs.cuda()

                        ins_outs = torch.cat((ins_outs, ins_out), 0)
                        del instance, ins_out

                        if idx >= opt.max_sentence_in_bag:
                            break

                
                bag_feature = selector_model(ins_outs)
                if opt.use_gpu: bag_feature = bag_feature.cuda()
                if (torch.max(bag_feature.squeeze(), 0)[1] == label):
                    train_cor += 1

                outs = torch.cat((outs, bag_feature), 0)
                del ins_outs, bag_feature

            # outs = outs.squeeze()
            # print("outs.size(): ", outs.size(), '\n', "labels.size(): ", labels.size())
            # print(outs,labels)
            loss = criterion(outs, labels)
            total_loss += loss.item()
            avg_loss = total_loss/(batch_num+1)
            sys.stdout.write("\rbatch number: {:6d}\tloss: {:7.4f}\ttrain_acc: {:7.2f}\t".format(batch_num, avg_loss, train_cor/len(labels)))
            sys.stdout.flush()
            # sys.stdout.write('\033')

            loss.backward()
            if opt.encoder == 'BERT':
                scheduler.step()
            optimizer.step()
            del outs, labels

        if (opt.skip_predict!=True):
            with torch.no_grad():
                predict(encoder_model, selector_model, test_data_loader)

    t = time.strftime('%m_%d_%H_%M.pth')
    torch.save(encoder_model.state_dict(), 'checkpoints/{}_{}'.format(opt.encoder, t))
    torch.save(selector_model.state_dict(), 'checkpoints/{}_{}'.format(opt.selector, t))

def predict(encoder_model, selector_model, test_data_loader):
    encoder_model.eval()
    selector_model.eval()
    y_true = []
    y_pred = []
    # confusion_matrix = [[0]*opt.rel_num]*opt.rel_num
    max_pre = 0.0
    max_rec = 0.0
    pred_res = []
    for epoch in range(1):
        test_cor = 0
        total_sample = 0
        # many bags in one batch(I guess...)
        
        count48 = 0
        for batch_num, (data, label_set) in enumerate(test_data_loader):
            # if (batch_num>3000):
            #     break
            # label_set is the label of each bag (there may be no more than 4 labels, but we only wants the first)

            # labels = []
            outs = torch.empty(0)
            # for l in label_set:
            #     if (l[0] != 0):
            #         labels.append(l[0])

            if opt.use_gpu:
                # labels = torch.LongTensor(labels).cuda()
                outs = outs.cuda()
            # else:
            #     labels = torch.LongTensor(labels)
            
            for idx, bag in enumerate(data):
                if (label_set[idx][0] == 0):
                    continue
                insNum = bag[1]
                label = label_set[idx][0]
                ins_outs = torch.empty(0)
                instances = bag[2]

                pf_list = []
                mask_list = []
                if opt.encoder!='BERT':
                    pf_list = bag[3]
                    mask_list = bag[5]

                # pf_list = bag[3]
                ins_out = torch.empty(0)
                encoder_model.batch_size = insNum
                if opt.use_gpu:
                    instances = torch.LongTensor(instances).cuda()

                if opt.encoder == 'BERT':
                    # with torch.no_grad():
                        # print(instances.size(0))
                    if insNum > opt.max_sentence_in_bag:
                        ins_outs = encoder_model(instances[:opt.max_sentence_in_bag])[0]
                    else:
                        ins_outs = encoder_model(instances)[0]
                else:

                    for idx, instance in enumerate(instances):
                        if opt.use_gpu:
                            pfs = torch.LongTensor(pf_list[idx]).cuda()
                            masks = torch.LongTensor(mask_list[idx]).cuda()
                        else:
                            pfs = torch.LongTensor(pf_list[idx])
                            masks = torch.LongTensor(mask_list[idx])

                        if opt.encoder=='PCNN':
                            ins_out = encoder_model(instance, pfs, masks)
                        else:
                            ins_out = encoder_model(instance, pfs)

                        if (opt.use_gpu):
                            ins_out = ins_out.cuda()
                            ins_outs = ins_outs.cuda()

                        ins_outs = torch.cat((ins_outs, ins_out), 0)
                        del instance, ins_out

                        if idx >= opt.max_sentence_in_bag:
                            break
                bag_feature = selector_model(ins_outs)
                if opt.use_gpu: bag_feature = bag_feature.cuda()
                prob, p_label = torch.max(bag_feature.squeeze(), 0)
                if p_label.item() == 48:
                    count48 += 1
                # confusion_matrix[p_label][label] += 1
                y_true.append(label)
                y_pred.append(p_label.item())
                pred_res.append([label, p_label.item(), prob.item()])
                total_sample += 1
                # print("label: {}, p_label: {}".format(label, p_label))
                if (p_label == label):
                    test_cor += 1
                outs = torch.cat((outs, bag_feature), 0)
        print("total samples: ", total_sample, "\tpred 48: ", count48)
        all_pre, all_rec = eval_metric_var(pred_res, total_sample)

        filename = opt.encoder+'_'+opt.selector+'_'+time.strftime('%m_%d_%H_%M')+'.txt'
        # save_pr(filename, all_pre=all_pre, all_rec=all_rec)
        prefix = 'out/'
        f = open(prefix+filename, 'w')
        for p, r in zip(all_pre, all_rec):
            f.write("{} {}\n".format(p, r))
        f.close()
    return

def save_pr(filename, all_pre, all_rec):
    prefix = 'out/'
    f = open(prefix+filename, 'w')
    for p, r in zip(all_pre, all_rec):
        f.write("{} {}\n".format(p, r))
    f.close()
    return

def eval_metric_var(pred_res, p_num):
    '''
    Apply the evalation method in  Lin 2016
    '''

    pred_res_sort = sorted(pred_res, key=lambda x: -x[2])
    correct = 0.0
    all_pre = []
    all_rec = []
    y_true = []
    y_pred = []
    labels = [i for i in range(opt.rel_num)]
    precision = 0.0
    recall = 0.0
    for i in range(len(pred_res_sort)):
        true_y = pred_res_sort[i][0]
        pred_y = pred_res_sort[i][1]
        y_true.append(true_y)
        y_pred.append(pred_y)
        if true_y == pred_y:
            correct += 1
        # Very rude way! Need to optimize!
        precision = metrics.precision_score(y_true, y_pred, labels=labels, average="weighted")
        recall = correct / p_num
        all_pre.append(precision)
        all_rec.append(recall)
    
    # metrics.confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    # If include NA, the p_num is not correct!
    print("\nPredict: test samples: {} acc: {} precision: {} recall: {}".format(p_num, accuracy, precision, recall) )
    # print("positive_num: {};  correct: {}".format(p_num, correct))
    return all_pre, all_rec

def test(**kwargs):
    opt.parse(kwargs)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    encoder_model = getattr(encoder_models, opt.encoder)(opt)
    opt.encoder_out_dimension = encoder_model.out_dimension
    selector_model = getattr(selector_models, opt.selector)(opt)

    prefix = 'checkpoints/'
    encoder_model.load_state_dict(torch.load(prefix+"EASY_07_12_01_27"+".pth", map_location='cuda:'+str(opt.gpu_id)))#EASY.pth
    
    selector_model.load_state_dict(torch.load(prefix+"ONE_07_12_01_27"+".pth", map_location='cuda:'+str(opt.gpu_id)))  # opt.selector AVG.pth

    if opt.use_gpu:
        encoder_model = encoder_model.cuda()
        selector_model = selector_model.cuda()
    
    DataModel = getattr(dataset, opt.data + 'Data')
    test_data = DataModel(opt.data_root, train=False)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    print('test data: {}'.format(len(test_data)))

    predict(encoder_model, selector_model, test_data_loader)

if __name__ == "__main__":
    import fire
    fire.Fire()
