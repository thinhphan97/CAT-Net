import random
from os.path import join
from random import randrange

import imageio.v2 as imageio
import jpegio
import numpy as np
import project_config
import torch
from PIL import Image
from Splicing.data.AbstractDataset import AbstractDataset


class TrainData(AbstractDataset):
    YCbCr2RGB = torch.tensor([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]], dtype=torch.float64)
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, read_from_jpeg=False, class_weight=None, mode ="train", val_num=300, train_ratio=[0.25, 0.25, 0.25, 0.25], train_num = 200000 ):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param tamp_list: EX: "Splicing/data/CASIA_list.txt"
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self._root_path = project_config.dataset_paths['COCO']
        self.read_from_jpeg = read_from_jpeg
        
        self._crop_size = crop_size
        self._grid_crop = grid_crop
        for block in blocks:
            assert block in ['RGB', 'DCTcoef', 'DCTvol', 'qtable']
        if 'DCTcoef' in blocks or 'DCTvol' in blocks or 'rawRGB' in blocks:
            assert grid_crop
        if grid_crop and crop_size is not None:
            assert crop_size[0] % 8 == 0 and crop_size[1] % 8 == 0
        self._blocks = blocks
        self.tamp_list = None
        self.DCT_channels = DCT_channels
        self.device = 'cpu'

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
            authentic_cls = [0] * len(authentic_names)
            splice_cls = [1] * len(splice_names)
            copymove_cls = [2] * len(copymove_names)
            inpainting_cls = [3] * len(inpainting_names)
            self.image_class = authentic_cls + splice_cls + copymove_cls + inpainting_cls
        
        
        self.train_num = train_num
        self.train_ratio = train_ratio
        self.mode = mode
        
        if class_weight is None:
            self.class_weights = torch.FloatTensor([1.0, 1.0])
        else:
            self.class_weights = torch.FloatTensor(class_weight)

    def _get_jpeg_info(self, im_path):
        """
        :param im_path: JPEG image path
        :return: DCT_coef (Y,Cb,Cr), qtables (Y,Cb,Cr)
        """
        num_channels = self.DCT_channels
        jpeg = jpegio.read(str(im_path))

        # determine which axes to up-sample
        ci = jpeg.comp_info
        need_scale = [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]
        if num_channels == 3:
            if ci[0].v_samp_factor == ci[1].v_samp_factor == ci[2].v_samp_factor:
                need_scale[0][0] = need_scale[1][0] = need_scale[2][0] = 2
            if ci[0].h_samp_factor == ci[1].h_samp_factor == ci[2].h_samp_factor:
                need_scale[0][1] = need_scale[1][1] = need_scale[2][1] = 2
        else:
            need_scale[0][0] = 2
            need_scale[0][1] = 2

        # up-sample DCT coefficients to match image size
        DCT_coef = []
        for i in range(num_channels):
            r, c = jpeg.coef_arrays[i].shape
            coef_view = jpeg.coef_arrays[i].reshape(r//8, 8, c//8, 8).transpose(0, 2, 1, 3)
            # case 1: row scale (O) and col scale (O)
            if need_scale[i][0]==1 and need_scale[i][1]==1:
                out_arr = np.zeros((r * 2, c * 2))
                out_view = out_arr.reshape(r * 2 // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
                out_view[::2, ::2, :, :] = coef_view[:, :, :, :]
                out_view[1::2, ::2, :, :] = coef_view[:, :, :, :]
                out_view[::2, 1::2, :, :] = coef_view[:, :, :, :]
                out_view[1::2, 1::2, :, :] = coef_view[:, :, :, :]

            # case 2: row scale (O) and col scale (X)
            elif need_scale[i][0]==1 and need_scale[i][1]==2:
                out_arr = np.zeros((r * 2, c))
                DCT_coef.append(out_arr)
                out_view = out_arr.reshape(r*2//8, 8, c // 8, 8).transpose(0, 2, 1, 3)
                out_view[::2, :, :, :] = coef_view[:, :, :, :]
                out_view[1::2, :, :, :] = coef_view[:, :, :, :]

            # case 3: row scale (X) and col scale (O)
            elif need_scale[i][0]==2 and need_scale[i][1]==1:
                out_arr = np.zeros((r, c * 2))
                out_view = out_arr.reshape(r // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
                out_view[:, ::2, :, :] = coef_view[:, :, :, :]
                out_view[:, 1::2, :, :] = coef_view[:, :, :, :]

            # case 4: row scale (X) and col scale (X)
            elif need_scale[i][0]==2 and need_scale[i][1]==2:
                out_arr = np.zeros((r, c))
                out_view = out_arr.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
                out_view[:, :, :, :] = coef_view[:, :, :, :]

            else:
                raise KeyError("Something wrong here.")

            DCT_coef.append(out_arr)

        # quantization tables
        qtables = [jpeg.quant_tables[ci[i].quant_tbl_no].astype(np.float) for i in range(num_channels)]

        return DCT_coef, qtables

    def _create_tensor(self, im_path, mask):
        ignore_index = -1

        img_RGB = np.array(Image.open(im_path).convert("RGB"))

        h, w = img_RGB.shape[0], img_RGB.shape[1]

        if 'DCTcoef' in self._blocks or 'DCTvol' in self._blocks or 'rawRGB' in self._blocks or 'qtable' in self._blocks:
            DCT_coef, qtables = self._get_jpeg_info(im_path)

        if mask is None:
            mask = np.zeros((h, w))

        if self._crop_size is None and self._grid_crop:
            crop_size = (-(-h//8) * 8, -(-w//8) * 8)  # smallest 8x8 grid crop that contains image
        elif self._crop_size is None and not self._grid_crop:
            crop_size = None  # use entire image! no crop, no pad, no DCTcoef or rawRGB
        else:
            crop_size = self._crop_size

        if crop_size is not None:
            # Pad if crop_size is larger than image size
            if h < crop_size[0] or w < crop_size[1]:
                # pad img_RGB
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1]), 3), 127.5)
                temp[:img_RGB.shape[0], :img_RGB.shape[1], :] = img_RGB
                img_RGB = temp

                # pad mask
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1])), ignore_index)  # pad with ignore_index(-1)
                temp[:mask.shape[0], :mask.shape[1]] = mask
                mask = temp

                # pad DCT_coef
                if 'DCTcoef' in self._blocks or 'DCTvol' in self._blocks or 'rawRGB' in self._blocks:
                    max_h = max(crop_size[0], max([DCT_coef[c].shape[0] for c in range(self.DCT_channels)]))
                    max_w = max(crop_size[1], max([DCT_coef[c].shape[1] for c in range(self.DCT_channels)]))
                    for i in range(self.DCT_channels):
                        temp = np.full((max_h, max_w), 0.0)  # pad with 0
                        temp[:DCT_coef[i].shape[0], :DCT_coef[i].shape[1]] = DCT_coef[i][:, :]
                        DCT_coef[i] = temp

            # Determine where to crop
            if self._grid_crop:
                s_r = (random.randint(0, max(h - crop_size[0], 0)) // 8) * 8
                s_c = (random.randint(0, max(w - crop_size[1], 0)) // 8) * 8
            else:
                s_r = random.randint(0, max(h - crop_size[0], 0))
                s_c = random.randint(0, max(w - crop_size[1], 0))

            # crop img_RGB
            img_RGB = img_RGB[s_r:s_r+crop_size[0], s_c:s_c+crop_size[1], :]

            # crop mask
            mask = mask[s_r:s_r + crop_size[0], s_c:s_c + crop_size[1]]

            # crop DCT_coef
            if 'DCTcoef' in self._blocks or 'DCTvol' in self._blocks or 'rawRGB' in self._blocks:
                for i in range(self.DCT_channels):
                    DCT_coef[i] = DCT_coef[i][s_r:s_r+crop_size[0], s_c:s_c+crop_size[1]]
                t_DCT_coef = torch.tensor(DCT_coef, dtype=torch.float) # final (but used below)

        # # handle 'RGB'
        if 'RGB' in self._blocks:
            t_RGB = (torch.tensor(img_RGB.transpose(2,0,1), dtype=torch.float)-127.5)/127.5  # final
        #     # t_RGB = t_RGB
            
        # # handle 'DCTvol'
        if 'DCTvol' in self._blocks:
            T = 20
            t_DCT_vol = torch.zeros(size=(T+1, t_DCT_coef.shape[1], t_DCT_coef.shape[2]))
            t_DCT_vol[0] += (t_DCT_coef == 0).float().squeeze()
            for i in range(1, T):
                t_DCT_vol[i] += (t_DCT_coef == i).float().squeeze()
                t_DCT_vol[i] += (t_DCT_coef == -i).float().squeeze()
            t_DCT_vol[T] += (t_DCT_coef >= T).float().squeeze()
            t_DCT_vol[T] += (t_DCT_coef <= -T).float().squeeze()

        # create tensor
        # img_block = []
        # for i in range(len(self._blocks)):
        #     if self._blocks[i] == 'RGB':
        #         img_block.append(t_RGB)
        #     elif self._blocks[i] == 'DCTcoef':
        #         img_block.append(t_DCT_coef)
        #     elif self._blocks[i] == 'DCTvol':
        #         img_block.append(t_DCT_vol)
        #     elif self._blocks[i] == 'qtable':
        #         continue
        #     else:
        #         raise KeyError("We cannot reach here. Something is wrong.")

        # # final tensor
        # tensor = torch.cat(img_block, dim=0)
    
        # if 'qtable' not in self._blocks:
        #     return tensor, torch.tensor(mask, dtype=torch.long), 0
        # else:
        return t_RGB, t_DCT_vol, torch.tensor(mask, dtype=torch.long), torch.tensor(qtables[:self.DCT_channels], dtype=torch.float)


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
        
    #     return Y_qtable
    
    def __len__(self):
        if self.mode == "train":
            return sum([len(item) for item in self.image_names])
            # return self.train_num
        else:
            return len(self.image_names)
        
    def __getitem__(self, index):
        
        # return self.get_tamp(index)
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
            del one_cls_names
        else:
            image_name = self.image_names[index]
            cls = self.image_class[index]
            
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

        # self._create_tensor(image_name, mask)
        # return mask, mask, mask

        return self._create_tensor(image_name, mask)
    
    def shuffle(self):
        
        for image_names in self.image_names:
            random.shuffle(image_names)
