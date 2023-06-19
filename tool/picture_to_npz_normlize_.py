# 2023.4.10 杨鹏 #图片，标签和边缘转成npz.
import cv2
import numpy as np
import os

def npz():
    #图像路径
    path111 = r'F:\third_paper\filaments-unet-dataset\test\img_640/'
    dirs = os.listdir(path111)
    #项目中存放训练所用的npz文件路径
    path2 = r'F:\third_paper\filaments-unet-dataset\test\npz\\'

    for file in dirs:
    	#读入图像
        image_path=(path111+file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #读入标签
        label_path = (r'F:\third_paper\filaments-unet-dataset\mask_1_255_1/'+file)
        label = cv2.imread(label_path, flags=0)
        label_path1 = (r'F:\third_paper\filaments-unet-dataset\canny_edge/'+file)
        edge = cv2.imread(label_path1, flags=0)
		#保存npz
        np.savez(path2+file[0:-4],image=image,label=label,edge=edge)
        print('------------',file,label_path,label_path1)


    print('npz——ok')


if __name__ == "__main__":
    # npz()
    with open(r'F:\third_paper\filaments-unet-dataset\test.txt', 'w') as f:
        for img_name in range(1,100+1):
            # f.write(str('Misc_')+str(img_name) + '\n')  # -5表示从后往前数，到小数点位置
            f.write(str(img_name) + '\n')  # -5表示从后往前数，到小数点位置

    print('文件名——ok')
