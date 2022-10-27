from os.path import join
from random import randrange

import imageio.v2 as imageio
import torch
import numpy as np
import project_config
from Splicing.data.AbstractDataset import AbstractDataset
import random

class TrainData(AbstractDataset):
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, read_from_jpeg=False, class_weight=None, mode ="train", val_num=300, train_ratio=[0.25, 0.25, 0.25, 0.25], train_num = 100000 ):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/CASIA_list.txt"
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self._root_path = project_config.dataset_paths['COCO']
        self.read_from_jpeg = read_from_jpeg
        
        # authentic
        authentic_names = []
        authentic_path = join(self._root_path, 'authentic/authentic')

        with open(join(authentic_path, 'authentic.txt')) as f:
            contents = f.readlines()
            if mode == "train":
                for content in contents[val_num:]:
                    authentic_names.append(join(authentic_path, content.strip()))
            else:
                for content in contents[:val_num]:
                    authentic_names.append(join(authentic_path, content.strip()))

        # splice
        splice_names = []
        splice_path = join(self._root_path, 'splice/splice')

        with open(join(splice_path, 'fake.txt')) as f:
            contents = f.readlines()
            if mode == "train":
                for content in contents[val_num:]:
                    splice_names.append(join(splice_path, content.strip()))
            else:
                for content in contents[:val_num]:
                    splice_names.append(join(splice_path, content.strip())) 
                
        splice_randmask = []
        splice_randmask_path = join(self._root_path, 'splice_randmask/splice_randmask')

        with open(join(splice_randmask_path, 'fake.txt')) as f:
            contents = f.readlines()
            if mode == "train":
                for content in contents[val_num:]:
                    splice_randmask.append(join(splice_randmask_path, content.strip()))
            else:
                for content in contents[:val_num]:
                    splice_randmask.append(join(splice_randmask_path, content.strip()))
                    
        splice_names = splice_names + splice_randmask

        # copymove
        copymove_names = []
        copymove_path = join(self._root_path, 'copymove/copymove')

        with open(join(copymove_path, 'fake.txt')) as f:
            contents = f.readlines()
            if mode == "train":
                for content in contents[val_num:]:
                    copymove_names.append(join(copymove_path, content.strip().replace(".png",".jpg")))
            else:
                for content in contents[:val_num]:
                    copymove_names.append(join(copymove_path, content.strip().replace(".png",".jpg")))
                    
        # inpainting
        inpainting_names = []
        inpainting_path = join(self._root_path, 'removal/removal')

        with open(join(inpainting_path, 'fake.txt')) as f:
            contents = f.readlines()
            if mode == "train":
                for content in contents[val_num:]:
                    inpainting_names.append(join(inpainting_path, content.strip()))
            else:
                for content in contents[:val_num]:
                    inpainting_names.append(join(inpainting_path, content.strip()))
                   
        if mode == "train":
            self.image_names = [authentic_names, splice_names, copymove_names, inpainting_names]
#             self.image_names = authentic_names + splice_names + copymove_names + inpainting_names
        else:
            self.image_names = authentic_names + splice_names + copymove_names + inpainting_names
        
        
        
        self.train_num = train_num
        self.train_ratio = train_ratio
        self.mode = mode
        
        if class_weight is None:
            self.class_weights = torch.FloatTensor([1.0, 1.0])
        else:
            self.class_weights = torch.FloatTensor(class_weight)

    def get_tamp(self, index):
                
        train_num = self.train_num
        train_ratio = self.train_ratio
        
        # get 4 class
        if self.mode == "train":
            if index < train_num * train_ratio[0]:
                cls = 0
            elif train_num * train_ratio[0] <= index < train_num * (train_ratio[0] + train_ratio[1]):
                cls = 1
            elif train_num * (train_ratio[0] + train_ratio[1]) <= index < train_num * (
                    train_ratio[0] + train_ratio[1] + train_ratio[2]):
                cls = 2
            else:
                cls = 3
            
            # get images in that class
            one_cls_names = self.image_names[cls]
            
            index = randrange(0, len(one_cls_names))

            # read the chosen image
            image_name = one_cls_names[index]
        else:
            image_name = self.image_names[index]
            
        image = imageio.imread(image_name)

        im_height, im_width, im_channel = image.shape
        
         # authentic
        if cls == 0:
            mask = np.zeros((im_height, im_width)).astype(np.uint8)

        # splice
        elif cls == 1:
            # if '.jpg' in image_name:
            mask_name = image_name.replace('fake', 'mask').replace('.jpg', '.png')
            # else:
            #     mask_name = image_name.replace('fake', 'mask').replace('.tif', '.png')
            mask = imageio.imread(mask_name)
            mask = np.asarray(mask)

        # copymove
        elif cls == 2:
            mask = imageio.imread(image_name.replace('fake', 'mask').replace('.jpg', '.png'))
            mask = np.asarray(mask)
        # inpainting
        elif cls == 3:
            mask = imageio.imread(image_name.replace('fake', 'mask').replace('.jpg', '.png'))
            mask = np.asarray(mask)

        else:
            raise Exception('class is not defined!')
        
        mask[mask > 0] = 1
        
        return self._create_tensor(image_name, mask) 


    def get_qtable(self, index):
        
        train_num = self.train_num
        train_ratio = self.train_ratio
        
        # get 4 class
        if index < train_num * train_ratio[0]:
            cls = 0
        elif train_num * train_ratio[0] <= index < train_num * (train_ratio[0] + train_ratio[1]):
            cls = 1
        elif train_num * (train_ratio[0] + train_ratio[1]) <= index < train_num * (
                train_ratio[0] + train_ratio[1] + train_ratio[2]):
            cls = 2
        else:
            cls = 3
        
         # get images in that class
        one_cls_names = self.image_names[cls]

        index = randrange(0, len(one_cls_names))

        # read the chosen image
        image_name = one_cls_names[index]
        
        if not str(image_name).lower().endswith('.jpg'):
            return None
        DCT_coef, qtables = self._get_jpeg_info(image_name)
        Y_qtable = qtables[0]
        
        return Y_qtable
    
    def __len__(self):
        if self.mode == "train":
            return self.train_num
        else:
            return len(self.image_names)
        
    def __getitem__(self, index):
        
        return self.get_tamp(index)