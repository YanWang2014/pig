# https://github.com/pytorch/vision/blob/master/torchvision/transforms.py
# https://github.com/ncullen93/torchsample/tree/master/torchsample/transforms
# https://keras.io/preprocessing/image/
'''
https://zhuanlan.zhihu.com/p/29513760

resize
rescale
noise
flip
rotate
shift
zoom
shear
contrast
channel shift
PCA
gamma
'''

import torch
from torchvision import transforms
import random
from PIL import Image
from .transforms_master import ColorJitter, scale, ten_crop, to_tensor

#input_size = 224 
#train_scale = 256 
#test_scale = 256
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def my_transform(img, input_size, train_scale, test_scale):
    img = scale(img, test_scale)
    imgs = ten_crop(img, input_size)  # this is a list of PIL Images
    return torch.stack([normalize(to_tensor(x)) for x in imgs], 0) # returns a 4D tensor
class my_ten_crops(object):
    def __init__(self, input_size, train_scale, test_scale):
        self.input_size = input_size
        self.train_scale = train_scale
        self.test_scale = test_scale
    def __call__(self, img):
        return my_transform(img, self.input_size, self.train_scale, self.test_scale)

# following ResNet paper, note that center crop should be removed if we can handle different image sizes in a batch
def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)
class HorizontalFlip(object):
    def __init__(self, flip_flag):
        self.flip_flag = flip_flag
    def __call__(self, img):
        if self.flip_flag:
            return hflip(img)
        else:
            return img
def my_transform_multiscale_test(varied_scale, flip_flag):  
    return transforms.Compose([
        transforms.Scale(varied_scale),  
        transforms.CenterCrop(varied_scale), 
        HorizontalFlip(flip_flag),
        transforms.ToTensor(),
        normalize
    ])

composed_data_transforms = {}
def data_transforms(phase, input_size = 224, train_scale = 256, test_scale = 256):
    print('input_size %d, train_scale %d, test_scale %d' %(input_size,train_scale,test_scale))
    
    composed_data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(input_size), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        normalize
    ]),
    'train2': transforms.Compose([
        transforms.RandomSizedCrop(input_size), 
        transforms.RandomHorizontalFlip(), 
        ColorJitter(),
        transforms.ToTensor(), 
        normalize
    ]),
    'multi_scale_train': transforms.Compose([   ## following ResNet paper, but not include the standard color augmentation from AlexNet
        transforms.Scale(random.randint(384, 640)),  # May be adjusted to be bigger
        transforms.RandomCrop(input_size),  # not RandomSizedCrop
        transforms.RandomHorizontalFlip(), 
        ColorJitter(), # different from AlexNet's PCA method which is adopted in the ResNet paper?
        transforms.ToTensor(), 
        normalize
    ]),
    'validation': transforms.Compose([
        transforms.Scale(test_scale),  
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Scale(test_scale),  
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'ten_crop': my_ten_crops(input_size, train_scale, test_scale)#todo: merge my_transform
    }
    return composed_data_transforms[phase]
