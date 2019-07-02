import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from collections import defaultdict

from utils.progress import print_progress, progress
from utils.data_downloader import downloder


class Dataset_Base():
    def __init__(self, gpu_num, batch_size, use_lang, shuffle):
        self._batch_size = batch_size
        self.gpu_num = gpu_num
        self.use_lang = use_lang
        self.shuffle = shuffle

        self.now_iter = 0
        self.now_epoch = 0
        self.now_loc = 0

        self.en_index2tok = None;self.jp_index2tok = None
        self.train_id_list = None;self.val_id_list = None
        self.train_en_caption = None;self.val_en_caption = None
        self.train_jp_caption = None;self.val_jp_caption = None
        self.train_image_data = None;self.val_image_data = None
        
    
    def get_data(self, is_val=False):
        if not is_val:
            use_data = "train"
            self.now_epoch += 1
        else:use_data = "val"

        id_list = getattr(self, f"{use_data}_id_list")
        image_data = getattr(self, f"{use_data}_image_data")
        if self.use_lang == "jp":
            caption = getattr(self, f"{use_data}_jp_caption")
        elif  self.use_lang == "en":
            caption = getattr(self, f"{use_data}_en_caption")

        self.now_loc = 0

        for i in range(0, len(id_list), self.batch_size):
            batch = id_list[i:i+self._batch_size]
            if len(batch) % 2 != 0:
                batch += [random.choice(id_list)]
            image = [np.load(image_data[key]) for key in batch]
            text = [random.choice(caption[key]) for key in batch]
            
            wrong = random.sample(id_list[:self.now_loc] + id_list[self.now_loc+self._batch_size:], len(batch))
            wrong_text = [random.choice(caption[key]) for key in wrong]

            self.now_loc += self._batch_size
            self.now_iter += 1

            yield image, text, wrong_text
                        
    def path2array(self, path):
        try:
            encoder = getattr(self, "encoder")
        except:
            self.encoder = models.vgg16_bn(pretrained=True)
            features = list(self.encoder.classifier.children())[:-1]
            self.encoder.classifier = nn.Sequential(*features)
            if torch.cuda.is_available():self.encoder.cuda()
            encoder = self.encoder
        
        encoder.eval()

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        def convert_img(img):
            img = img.transpose(2,0,1).astype("float32") / 255 - 0.5
            if torch.cuda.is_available():
                return torch.Tensor(img).view(1,3,224,224).cuda()
            else:
                return torch.Tensor(img).view(1,3,224,224)

        img_feat = encoder(convert_img(img))

        if torch.cuda.is_available():
            img_feat = img_feat.data.cpu().numpy()
        else:
            img_feat = img_feat.data.numpy()
            
        return img_feat[0]

    def sample(self, sample_size):
        if self.use_lang == "jp":
            caption = getattr(self, "train_jp_caption")
        elif  self.use_lang == "en":
            caption = getattr(self, "train_en_caption")

        sample = random.sample(self.train_id_list, sample_size)

        image = [np.load(self.train_image_data[key]) for key in sample]
        text = [random.choice(caption[key]) for key in sample]

        return sample, image, text

    def clear_state(self):
        self.now_iter = 0
        self.now_epoch = 0
        self.now_loc = 0
    
    @property
    def get_state(self):
        return self.now_epoch, self.now_iter, self.now_loc
        
    @property
    def en_voc_size(self):
        return len(list(self.en_index2tok.keys()))
    
    @property
    def jp_voc_size(self):
        return len(list(self.jp_index2tok.keys()))
        

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        self._batch_size = new_batch_size

    @property
    def train_data_len(self):
        return len(self.train_id_list)

    @property
    def val_data_len(self):
        return len(self.val_id_list)