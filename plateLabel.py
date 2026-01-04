import cv2
# import imageio
import numpy as np
import os
import shutil
import argparse
from tqdm import tqdm
from alphabets import plate_chr


def allFileList(rootfile,allFile):
    folder =os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile,temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName,allFile)


def is_str_right(plate_name):
    for str_ in plate_name:
        if str_ not in palteStr:
            return False
    return True


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="/mnt/Gu/trainData/plate/final", help='source') 
    parser.add_argument('--label_file', type=str, default='datasets/train.txt', help='model.pt path(s)')  
    
    opt = parser.parse_args()
    rootPath = opt.image_path
    labelFile = opt.label_file
    # palteStr=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民深危险品0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    # palteStr=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航深0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    palteStr=plate_chr
    print(len(palteStr))
    plateDict ={}
    for i in range(len(list(palteStr))):
        plateDict[palteStr[i]]=i
    fp = open(labelFile, "w", encoding="utf-8")
    file =[]
    allFileList(rootPath,file)
    print(f'total image num: {len(file)}')

    picNum = 0
    for jpgFile in tqdm(file, desc="Processing images", ncols=100):
        # print(jpgFile)
        jpgName = os.path.basename(jpgFile)
        name = jpgName.split("_")[0]
        if " " in name:
            continue
        if not is_str_right(name):
            continue

        labelStr = " "
        strList = list(name)
        for char in strList:
            labelStr += str(plateDict[char]) + " "

        picNum += 1
        fp.write(jpgFile + labelStr + "\n")
    fp.close()
    print(f'valid image num: {picNum}')
