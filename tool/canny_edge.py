#  -*- coding: utf-8 -*-
import cv2
import os
import glob

import cv2
import numpy as np

def Edge_Extract(root):
    #在root下面建立两个文件夹，masks用来保存自己已有的二值化图像，edge用来保存自己生成的轮廓
    img_root = os.path.join(root,'mask_1_255')			# 修改为保存图像的文件名
    edge_root = os.path.join(root,'canny_edge')			# 结果输出文件

    if not os.path.exists(edge_root):
        os.mkdir(edge_root)

    file_names = os.listdir(img_root)
    img_name = []

    for name in file_names:
        if not name.endswith('.png'):
            assert "This file %s is not PNG"%(name)
        print(os.path.join(img_root,name[:-4]+'.png'))
        img_name.append(os.path.join(img_root,name[:-4]+'.png'))

    index = 0
    for image in img_name:
        img = cv2.imread(image,0)
        cv2.imwrite(edge_root+'/'+file_names[index],cv2.Canny(img,0,200))
        index += 1
    return 0

if __name__ == '__main__':
    root = r'F:\third_paper\filaments-unet-dataset/'          # 修改为w对应的文件路径
    Edge_Extract(root)
