import matplotlib.pyplot as plt

# files = ['EASY_ONE_07_12_01_27.txt', 'EASY_AVG_07_11_16_47.txt', 'EASY_ATT_07_13_17_12.txt']
files = [#'EASY_ATT_07_13_21_18.txt', 'BERT_ONE_08_08_22_39.txt','BERT_ONE_08_08_23_10.txt', 'BERT_ONE_08_09_01_53.txt', 'BERT_ONE_08_09_02_49.txt', 'EASY_ONE_08_09_19_41.txt','EASY_ONE_08_08_16_20.txt', \
     'EASY_ONE_07_13_21_28.txt', 'BERT_ONE_08_09_04_12.txt', 'EASY_ONE_08_10_00_48.txt', 'BERT_ONE_08_12_07_08.txt']

names = ['CNN(original)', 'CNN', 'BERT', 'BERT(with position)']
plts = []

plt.figure()
plt.ylim(0,1.1)
plt.xlabel("Recall")
plt.xlim(0,1.1)
plt.ylabel("Precison")

for filename in files:
    f = open('out/'+filename, 'r')
    all_pred = []
    all_rec = []
    for line in f.readlines():
        pred, rec = line.strip('\n').split(' ')
        all_pred.append(float(pred))
        all_rec.append(float(rec))
    p, = plt.plot(all_rec, all_pred, label='line')
    plts.append(p)
    f.close()
plt.legend(plts, names, loc="lower right") #files

plt.show()
