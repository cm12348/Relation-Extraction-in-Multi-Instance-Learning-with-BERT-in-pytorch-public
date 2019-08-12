# -*- coding: utf-8 -*-
import torch
import numpy as np
# from pytorch_pretrained_bert import BertTokenizer
from pytorch_transformers import BertTokenizer
import os
import sys

tokenizer = BertTokenizer.from_pretrained('./downloaded_weights/downloaded_bert_base_uncased')

wordlist = []

f = open("dataset/NYT/vector.txt", 'r', encoding="utf-8")
for line in f:
    line = line.strip('\n').split()
    wordlist.append(line[0])
id2word = {i: j for i, j in enumerate(wordlist)}

max_id = 0
for mode in ['train', 'test']:
    path = os.path.join("dataset/NYT", mode + "/")

    labels = np.load(path + 'labels.npy')
    bag_feature = np.load(path + 'bags_feature.npy')
    for idx, bag in enumerate(bag_feature):
        sentences = []
        # print(bag[2])
        # entities = bag[0]
        # for i in range(5):
        #     print('bag'+str(i)+": ", bag[i])
        # print(bag)
        pos = bag[4]
        for ins_id, instance in enumerate(bag[2]):
            # print(len(instance))
            # print(instance, pos)
            sentence = ""
            for p, word_id in enumerate(instance):
                word = id2word[word_id]

                if (p+1 in pos[ins_id]):
                    word = '#' + word + '#' #加不加空格是一样的
                if (word == 'BLANK'):
                    word = ' '
                sentence += word + ' '
            # print(sentence)
            tokenized_text = tokenizer.tokenize(sentence)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

            # The default tokenization fix sentence length to 82
            length = len(indexed_tokens)
            if length < 82:
                for i in range(82 - length):
                    indexed_tokens.append(0)
            elif length > 82:
                indexed_tokens = indexed_tokens[:82]
            # print(len(indexed_tokens))
            max_id = max(indexed_tokens)
            sentences.append(indexed_tokens)
            # print(len(sentence))
        bag_feature[idx][2] = sentences
        # print(bag_feature[idx][2])
        every = 100
        if idx % every == 0:
            if mode == 'train':
                sys.stdout.write("\r{} data: {:.4}%\tmax: {}".format(mode, (idx*every)/(293175), max_id))
            else:
                sys.stdout.write("\r{} data: {:.4}%\tmax: {}".format(mode, (idx*every)/(96678), max_id))
            sys.stdout.flush()
        
    np.save(os.path.join('dataset/NYT', mode, 'new_bert_bags_feature.npy'), bag_feature)
#     print(bag[2])
# print(type(labels))
