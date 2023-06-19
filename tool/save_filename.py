# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os

img_path = r'F:\TransUNet-main\TransUNet-main\data\MBP\train_npz/'
img_list = os.listdir(img_path)
print('img_list: ', img_list)

with open(r'F:\TransUNet-main\TransUNet-main\lists\lists_MBP\train.txt', 'w') as f:
    for img_name in img_list:
        f.write(img_name[0:-4] + '\n')  # -5表示从后往前数，到小数点位置
        # f.write(img_name + '\n')