import matplotlib.pyplot as plt
import cv2
import numpy as np

if __name__ == '__main__':
    img = '/media/xwj/xwjdata/Dataset/ShanghaiTech/original/part_A/train_data/images/IMG_86.jpg'
    im = cv2.imread(img)
    im = im.transpose(2,0,1)
    imt = im[:,:,::-1]
    plt.imshow(imt)
    plt.show()