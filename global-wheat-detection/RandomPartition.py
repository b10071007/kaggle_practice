# -*- coding: utf-8 -*-

import numpy as np

def outputListFile(fList, out_dir):
    with open(out_dir,'w') as fObj_w:
        for line in fList:
            fObj_w.writelines(line)
    
rootPath = "D:/Dataset/kaggle/global-wheat-detection/"
src_mapList = rootPath + "train.txt"
out_trainList = rootPath + "_train_map.txt"
out_valList = rootPath + "_val_map.txt"

train_ratio = 0.8

print("Random sample {} as training set".format(train_ratio))
with open(src_mapList, 'r') as fObj_r:
    mapList = fObj_r.readlines()
  
np.random.seed(1)

np.random.shuffle(mapList)
np.random.shuffle(mapList)
np.random.shuffle(mapList)

num_total = len(mapList)
num_train = int(num_total*train_ratio)

train_List = mapList[:num_train]
val_List = mapList[num_train:]

outputListFile(train_List, out_trainList)
outputListFile(val_List, out_valList)


