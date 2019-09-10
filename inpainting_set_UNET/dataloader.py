import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image
import glob

from torchvision import transforms

class DataLoader():
    
    def __init__(self, root_dir='data', batch_size=16, no_epochs = 1):
        self.batch_size = batch_size
        self.no_epochs = no_epochs
        
        self.root_dir = abspath(root_dir)
        self.train_img_path = join(self.root_dir, 'train.png')
        self.test_img_path = join(self.root_dir, 'test.png')

    def __iter__(self):

        if self.mode == 'train':
            no_epochs = 100
            img_path = self.train_img_path
            crop_width = 450
            endId = no_epochs

        elif self.mode == 'test':
            no_epochs = 1
            img_path = self.test_img_path
            crop_width = 30
            endId = no_epochs

        current = 0
        while current < endId:

            test_image = Image.open(img_path)
            data_list = list()
            gt_list = list()

            data_list.clear()
            gt_list.clear()
            for i in range(self.batch_size):

                crop_width = 350
                crop_height = 350
                crop_size = 350
                
                crop_x = random.randint(0,test_image.height - crop_height)
                crop_y = random.randint(0,test_image.width - crop_width)
                
                data_image = transforms.functional.resized_crop(test_image, crop_x, crop_y, crop_width, crop_height, crop_size, Image.BILINEAR)

                if self.mode == 'train':
                    if random.random() > 0.5:
                        
                        #print("1")
                        hue_factor = random.randint(1, 5) / 10
                        data_image = transforms.functional.adjust_hue(data_image, hue_factor)

                        gamma = random.randint(10, 12) / 10
                        data_image = transforms.functional.adjust_gamma(data_image, gamma, gain=1)
                    
                    # if random.random() > 5:
                    #     i = random.randint(0, data_image.shape[1] / 2)
                    #     j = random.randint(0, data_image.shape[0] / 2)
                    #     h = random.randint(20, data_image.shape[1] - i)
                    #     w = random.randint(20, data_image.shape[0] - j)
                    #     data_image = transforms.functional.crop(data_image, i, j, h, w)
                    #     label_image = transforms.functional.crop(label_image, i, j, h, w)
                    
                    if random.random() > 0.5:
                        
                        #print("2")
                        data_image = transforms.functional.hflip(data_image)

                    if random.random() > 0.5:
                        
                        #print("3")
                        data_image = transforms.functional.vflip(data_image)

                    if random.random() > 0.5:

                        #print("4")
                        data_image = transforms.functional.rotate(data_image, random.randint(-45, 45))
                        data_image = np.asarray(transforms.functional.five_crop(data_image, 150)[4])
                        data_image = Image.fromarray(data_image)

                    if random.random() > 0.5:

                        #print("5")
                        data_image = transforms.functional.resize(data_image, random.randint(200,500)) 

                data_image = data_image.resize((128,128),Image.ANTIALIAS)
                gt_image = np.array(data_image)
                mask = self.generateRandomMask (128, 64, 8)
                
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                data_image = (data_image * (mask / 255))

                temp_mask = mask[:, :, 0]
                temp_mask = np.expand_dims(temp_mask, axis=2)

                # making the data_image and mask of appropriate size
                data_image= np.concatenate((data_image, temp_mask), axis = 2)
                
                # append data image and mask to list
                data_list.append(data_image)
                gt_list.append(gt_image)

            data_list = np.array(data_list)
            gt_list = np.array(gt_list)

            data_list = data_list / 255
            gt_list = gt_list / 255
            
            current += 1

            yield (data_list, gt_list)

    def generateRandomMask (self, size, max_rec_width, max_rec_height):

        mask = np.full((size, size), 255)

        for i in range(5):

            if random.random() > 0.5:
                rec_x = random.randint(0,size - max_rec_width)
                rec_y = random.randint(0,size - max_rec_height)
                mask[rec_x:rec_x+max_rec_width, rec_y:rec_y+max_rec_height] = 0

            else:
                rec_x = random.randint(0,size - max_rec_height)
                rec_y = random.randint(0,size - max_rec_width)
                mask[rec_x:rec_x+max_rec_height, rec_y:rec_y+max_rec_width] = 0

        return mask

    def setMode(self, mode):
        self.mode = mode

    # def n_train(self):
    #     data_length = len(self.data_files)
    #     return np.int_(data_length - np.floor(data_length * self.test_percent))
        
