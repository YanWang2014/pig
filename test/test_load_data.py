# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:36:03 2017

@author: Beyond



http://www.scikit-video.org/stable/io.html

https://github.com/MohsenFayyaz89/PyTorch_Video_Dataset
http://blog.csdn.net/u011276025/article/details/76098185

https://stackoverflow.com/questions/42163058/how-to-turn-a-video-into-numpy-array
https://stackoverflow.com/questions/29718238/how-to-read-mp4-video-to-be-processed-by-scikit-image
https://gist.github.com/stokasto/1779208

两种读取思路(总之先试试用全图的，毕竟细粒度也是这样做的。后续可以尝试用detection把猪框出来，以去掉背景)：
用pims随机切片比较好？（结合class aware，另外通过设定帧数的范围来划分训练和验证集）
    https://arxiv.org/pdf/1512.05830.pdf
    http://scikit-image.org/docs/dev/user_guide/video.html
    http://soft-matter.github.io/pims/v0.4/video.html
用tf object detection找出猪的框，然后crop出来(虽然不认识猪，但是仍然可以当做bear或cow框出来==)
    https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/?completed=/introduction-use-tensorflow-object-detection-api-tutorial/
    Object Tracking in Tensorflow ( Localization Detection Classification ) developed to partecipate to ImageNET VID competition
        https://github.com/DrewNF/Tensorflow_Object_Tracking_Video
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
"""


import pims
from matplotlib import pyplot as plt

v = pims.PyAVVideoReader('../data/videos/1.mp4') # pims.pyav_reader.PyAVReaderTimed
#plt.imshow(v[-1])  # a 2D numpy array representing the last frame

Length = len(v)
#plt.figure()
#plt.imshow(v[3001])
#
#plt.figure()
#plt.imshow(v[30002])  # bug

#j = 0
#for i in v:
#    j += 1
#    print(j)

x = range(2000,2500,50) # 这样才60张图？感觉转换成图片不太行，还是用pims随机切片比较好？（结合class aware，另外通过设定帧数的范围来划分训练和验证集）
for i in x:
    plt.figure()
    plt.imshow(v[i])

#
#import shutil
#
#for i in range(1,31):
#    shutil.copyfile('../data/train_folder/'+str(i)+'/'+str(i)+'.mp4','../data/videos/'+str(i)+'.mp4')
    
    
class PigDataset(Dataset):

    def __init__(self, json_labels, root_dir, transform=None):
        self.label_raw = json_labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_raw)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.label_raw[idx]['image_id'])
        img_name_raw = self.label_raw[idx]['image_id']
        image = Image.open(img_name)
        label = int(self.label_raw[idx]['label_id'])

        if self.transform:
            image = self.transform(image)

        return image, label, img_name_raw