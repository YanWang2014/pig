'''
adapted from https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_image_loader.py


basic
    FaceNet: A Unified Embedding for Face Recognition and Clustering

怎么更好的分类
    【Embedding Label Structures for Fine-Grained Feature Representation】
    
怎么load数据
【总的load思路类似[3]。类内差异大的对策: 对于随机抽到的一个a，在它周围若干个帧(太小会过拟合？太大不靠谱，参见facenet图1)中抽一个p，然后在别的视频抽一个n。
抽完一个batch后用[2]的思路load图片】
【将这种类内差异大的想法扩展到分类和聚类上，其实我们最后希望得到的一个cluster代表的是某个单一拍摄角度下的同一只猪】
    [1] In Defense of the Triplet Loss for Person Re-Identification
        a generalization of the Lifted Embedding loss based on PK batches which considers all anchor-positive pairs 
    [2] https://github.com/andreasveit/triplet-network-pytorch
    [3] https://github.com/meismaomao/Pig2/tree/master/triplet_model

类内差异大(拍摄角度)
    NetVLAD: CNN architecture for weakly supervised place recognition
        a weakly supervised ranking loss
            https://zhuanlan.zhihu.com/p/22265265
其他参考
    https://github.com/davidsandberg/facenet/wiki/Triplet-loss-training
    https://discuss.pytorch.org/t/triplet-vs-cross-entropy-loss-for-multi-label-classification/4480
'''

from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import random
import json

class TripletImageDataset(torch.utils.data.Dataset):

    def __init__(self, json_labels, root_dir, transform, distance, frames):
        self.label_raw = json_labels
        self.root_dir = root_dir
        self.transform = transform
        self.distance = distance #10  #p在a的前后10帧中取
        self.classes = 30 # 共30类，存的label从0开始 
        self.frames = frames #561 #每个类别训练集不超过这个数

    def __len__(self):
        return len(self.label_raw)
    
    @staticmethod
    def default_image_loader(path):
        return Image.open(path) #.convert('RGB')
    
    @staticmethod
    def make_img_name(class_, frame_):
        return ('image%02d-%08d.jpg'%(class_, frame_))
    
    def pn_generator(self, a_name_raw):
        a_class = int(a_name_raw[5:7])
        a_frame = int(a_name_raw[8:16])
        
        while True:
            try:
#                print('try p')
                p_frame = random.randint(max(1,a_frame-self.distance), min(self.frames, a_frame+self.distance))
                assert p_frame != a_frame
                p_name = os.path.join(self.root_dir, self.make_img_name(a_class, p_frame))
                img_p = self.default_image_loader(p_name)
                break
            except (FileNotFoundError, AssertionError):
                continue
                   
        while True:
            try:
                n_class = random.randint(1, self.classes)   # 1-30
                assert n_class != a_class
                while True:
                    try:
#                        print('try n')
                        n_frame = random.randint(1, self.frames)
                        n_name = os.path.join(self.root_dir, self.make_img_name(n_class, n_frame))
                        img_n = self.default_image_loader(n_name)
                        return img_p, img_n
                    except FileNotFoundError:
                        continue
            except AssertionError:
                continue
        
    def __getitem__(self, idx):
#        if phases[0] == 'val':
#            img_name = self.root_dir+ '/' + str(self.label_raw[idx]['label_id']+1) + '/'+ self.label_raw[idx]['image_id']
#        else:
        img_name = os.path.join(self.root_dir, self.label_raw[idx]['image_id'])
        img_name_raw = self.label_raw[idx]['image_id']
        image = self.default_image_loader(img_name)
        label = self.label_raw[idx]['label_id']
        img_p, img_n = self.pn_generator(img_name_raw) # 依据a的类别和在视频中的位置生成p和n

        if self.transform:
            image = self.transform(image) # a
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        return image, img_p, img_n, label

#
#class TripletImageDataset(torch.utils.data.Dataset):
#    def __init__(self, base_path, filenames_filename, triplets_file_name, transform=None,
#                 loader=default_image_loader):
#        """ filenames_filename: A text file with each line containing the path to an image e.g.,
#                images/class1/sample.jpg
#            triplets_file_name: A text file with each line containing three integers, 
#                where integer i refers to the i-th image in the filenames file. 
#                For a line of intergers 'a b c', a triplet is defined such that image a is more 
#                similar to image c than it is to image b, e.g., 
#                0 2017 42 """
#        self.base_path = base_path  
#        self.filenamelist = []
#        for line in open(filenames_filename):
#            self.filenamelist.append(line.rstrip('\n'))
#        triplets = []
#        for line in open(triplets_file_name):
#            triplets.append((line.split()[0], line.split()[1], line.split()[2])) # anchor, far, close
#        self.triplets = triplets
#        self.transform = transform
#        self.loader = loader
#
#    def __getitem__(self, index):
#        path1, path2, path3 = self.triplets[index]
#        img1 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path1)]))
#        img2 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path2)]))
#        img3 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path3)]))
#        if self.transform is not None:
#            img1 = self.transform(img1)
#            img2 = self.transform(img2)
#            img3 = self.transform(img3)
#
#        return img1, img2, img3
#
#    def __len__(self):
#        return len(self.triplets)
