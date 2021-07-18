from torch.utils.data import Dataset
import numpy as np
from PIL import Image

import os


class FGDataset(Dataset):
    """
    Dataset Class for loading the data from ChangeDetection.Net format
    """
    def __init__(self, data_list, ip_path, gt_path, n):
        self.data_list = data_list
        self.ip_path = ip_path
        self.gt_path = gt_path
        
        self.multi_frame_data_list = list()
        
        for i in range(len(self.data_list)):
            low, high = i - n // 2, i + n // 2
            if low < 0 or high >= len(self.data_list):
                continue
            self.multi_frame_data_list.append(
                ([self.data_list[x][0] for x in range(i-n//2, i+n//2+1)], self.data_list[i][1])
            )
        
    def __len__(self):
        return len(self.multi_frame_data_list)
    
    def __getitem__(self, idx):
        ip_file_list, gt_file = self.multi_frame_data_list[idx]
        
        ip_imgs_scale_1 = self.get_ip_imgs_for_scale(ip_file_list, 240)
        ip_imgs_scale_2 = self.get_ip_imgs_for_scale(ip_file_list, 120)
        ip_imgs_scale_3 = self.get_ip_imgs_for_scale(ip_file_list, 60)
        
        ip_imgs = [ip_imgs_scale_1, ip_imgs_scale_2, ip_imgs_scale_3]
        
        gt_img_scale_1 = self.get_gt_img_for_scale(gt_file, 240)
        gt_img_scale_2 = self.get_gt_img_for_scale(gt_file, 120)
        gt_img_scale_3 = self.get_gt_img_for_scale(gt_file, 60)
        
        gt_img = [gt_img_scale_1, gt_img_scale_2, gt_img_scale_3]
    
        return ip_imgs, gt_img
    
    def get_ip_imgs_for_scale(self, ip_list, scale=240):
        ip_imgs = [np.array(Image.open(os.path.join(self.ip_path, f)).resize((scale,scale))) for f in ip_list]
        ip_imgs = np.concatenate(ip_imgs, axis=-1)
        ip_imgs = np.moveaxis(ip_imgs, -1, 0)
        ip_imgs = ((ip_imgs / 255) * 2) - 1
        return ip_imgs
    
    def get_gt_img_for_scale(self, gt_file, scale=240):
        gt_img = np.array(Image.open(os.path.join(self.gt_path, gt_file)).resize((scale,scale)).convert('L'))
        gt_img[gt_img < 255] = 0
        gt_img = gt_img / 255
        gt_img = gt_img.astype(int)
        gt_img = np.expand_dims(gt_img, axis=0)
        return gt_img
