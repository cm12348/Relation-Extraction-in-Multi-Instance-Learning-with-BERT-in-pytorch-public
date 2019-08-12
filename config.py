# -*- coding: utf-8 -*-

data_dic ={
    'NYT': {
        'data_root': './dataset/NYT/',
        'w2v_path': './dataset/NYT/w2v.npy',
        'p1_2v_path': './dataset/NYT/p1_2v.npy',
        'p2_2v_path': './dataset/NYT/p2_2v.npy',
        'vocab_size': 114043,
        'rel_num': 53
    },
    'FilterNYT': {
        'data_root': './dataset/FilterNYT/',
        'w2v_path': './dataset/FilterNYT/w2v.npy',
        'p1_2v_path': './dataset/FilterNYT/p1_2v.npy',
        'p2_2v_path': './dataset/FilterNYT/p2_2v.npy',
        'vocab_size': 160695 + 2,
        'rel_num': 27
    }
}

class DefaultConfig(object):
    batch_size = 4# 512
    data = 'NYT'
    drop_out = 0.5
    embed_dim = 50
    encoder = 'BERT'
    encoder_out_dimension = 30
    filters = [3]
    filters_num = 230
    filterwarning = 'default'
    gpu_id = 0
    lr = 0.01
    max_len = 80 + 2 # 2 padding. Max sentence length
    max_sentence_in_bag = 150
    norm_emb = True
    num_workers = 0
    num_epochs = 10
    num_filters = 30
    p1_2v_path = data_dic[data]['p1_2v_path']
    p2_2v_path = data_dic[data]['p2_2v_path']
    pos_size = 102
    pos_dim = 5
    rel_num = data_dic[data]['rel_num']
    selector = 'ONE'
    skip_predict = False
    use_bert_tokenizer = True # Set false to use the default tokenizer. Notice that Bert encoder must use Bert tokenizer.
    use_gpu = True
    use_pcnn = True
    vocab_size = data_dic[data]['vocab_size']
    w2v_path = data_dic[data]['w2v_path']
    weight_decay = 0.0001

def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)
        data_list = ['data_root', 'vocab_size']
        for r in data_list:
            setattr(self, r, data_dic[self.data][r])

        print('********************************************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('********************************************************************************')

DefaultConfig.parse = parse
opt = DefaultConfig()