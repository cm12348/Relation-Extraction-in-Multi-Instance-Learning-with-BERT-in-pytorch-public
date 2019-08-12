# README

- I refered many codes from [ShomyLiu's pytorch-relation-extraction](https://github.com/ShomyLiu/pytorch-relation-extraction), and refered the structure of [thunlp's OpenNRE](https://github.com/thunlp/OpenNRE) which is implemented in tensorflow. Thank them a lot!

## How to Run

- Environment: python3.5+, torch1.0.0+, pytorch_transformers.
- Data download: [百度网盘](https://pan.baidu.com/s/1Mu46NOtrrJhqN68s9WfLKg)  [Google Drive](https://drive.google.com/drive/folders/1kqHG0KszGhkyLA4AZSLZ2XZm9sxD8b58?usp=sharing). The preprocession is followed by [ShomyLiu](https://github.com/ShomyLiu/pytorch-relation-extraction). To get bert tokenized feature, you need to run 'dataset/generate_bert_feature.py' after the preprocession.
- Run "python3 main.py train". You can add parameters after this command like "python3 main.py train --batch_size=256 --data='NYT' --encoder='EASY'". Here is the command I used for training when BERT is encoder: "python main.py train --encoder='BERT' --selector='ONE' --use_gpu=True --gpu_id=0 --batch_size=16 --max_sentence_in_bag=10 --lr=0.001".
- To use BERT, you need firstly install pytorch_transformers. (reference: [huggingface's pytorch-pretrained-bert](https://github.com/huggingface/pytorch-transformers))

## Model Instruction

### Encoder Models

- Input: sentence representation with words' ids [1, sentence length] and position features [2, sentence length].
- Output: a sentence representation [1, dimension].
- Implemented: EASY(an easy CNN), BERT. (PCNN is not finished)

### Selector Models

- Input: encoder's outputs of a bag.
- Output: bag feature [1, relation number]
- Implemented: AVG, ONE, ATT(not sure if correct or not)
