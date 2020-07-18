# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

rootPath = "D:/Dataset/kaggle/global-wheat-detection/"
src_annotPath = rootPath + "train.csv"
out_annotPath = rootPath + "train.txt"

print("Read Annotation")
annotation = pd.read_csv(src_annotPath)

print("Parse Annotation")
num_annot = len(annotation)
imgList = np.array([]).astype(str)
annotDict = {}
for i in range(num_annot):
    image_id, width, height, bbox, _ = annotation.iloc[i]
    bbox = bbox[1:-1].split(',')
    bbox = np.array(bbox).astype(float).astype(int)
    bbox = np.append(bbox[0:2], 
                     [bbox[0]+bbox[2], bbox[1]+bbox[3]])

    if image_id not in imgList:
        imgList = np.append(imgList, image_id)
        annotDict[image_id] = " {},{},{},{},1".format(
                                bbox[0],bbox[1],bbox[2],bbox[3])
    else:
        annotDict[image_id] += " {},{},{},{},1".format(
                                bbox[0],bbox[1],bbox[2],bbox[3])

print("Save Preprocessed Annotation")
fObj_w = open(out_annotPath, "w")
for image_id in annotDict:
    box_str = annotDict[image_id]
    img_name = image_id + ".jpg"
    fObj_w.writelines(img_name + box_str + "\n")

fObj_w.close()   
    
    
