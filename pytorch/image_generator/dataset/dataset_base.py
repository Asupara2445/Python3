import os
import time
import glob
import json
import pickle
import random

import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

from utils.progress import print_progress, progress
from utils.data_downloader import downloder


def load_plk(path):
    with open(path,"rb") as fp:
        return pickle.load(fp)

class Dataset_Base():
    def __init__(self, gpu_num, batch_size, shuffle):
        self._batch_size = batch_size
        self.gpu_num = gpu_num
        self.shuffle = shuffle
        self.image_keys = ["x_64", "x_128", "x_256"]

        self.now_iter = 0
        self.now_epoch = 0
        self.now_loc = 0

        self.en_index2tok = None;self.jp_index2tok = None
        self.train_id_list = None;self.test_id_list = None;self.val_id_list = None
        self.train_image_data = None;self.train_en_caption = None
        self.val_image_data = None;self.val_en_caption = None
    
    def get_data(self, is_test=False):
        if not is_test:use_data = "train"
        else:use_data = "test"

        id_list = getattr(self, f"{use_data}_id_list")
        self.now_epoch += 1

        for i in range(0, self.train_data_len, self.batch_size):
            batch = id_list[i:i+self._batch_size]
            self.now_loc += self._batch_size

            image = [load_plk(self.train_image_data[key]) for key in batch]
            en_text = [random.choice(self.train_en_caption[key]) for key in batch]
            en_text_feature = [data[1] for data in en_text]


            wrong = id_list[self.now_loc:self.now_loc+self._batch_size]
            wrong_image = [load_plk(self.train_image_data[key]) for key in wrong]
            wrong_en_text = [random.choice(self.train_en_caption[key]) for key in wrong]
            wrong_en_text_feature = [data[1] for data in wrong_en_text]

            self.now_iter += 1

            yield image, en_text_feature, wrong_image, wrong_en_text_feature
                        

    def path2array(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_64, img_128, img_256 = cv2.resize(img, (64, 64)), cv2.resize(img, (128, 128)), cv2.resize(img, (256, 256))

        def convert_img(img):
            return img.transpose(2,0,1).astype("float32") / 255 - 0.5
        
        return convert_img(img_64), convert_img(img_128), convert_img(img_256)

    def sampling(self, size=10):
        sample =  random.sample(self.test_id_list, size)
        
        image_sample = [load_plk(self.train_image_data[key]) for key in sample]
        en_text_sample =  [random.choice(self.train_en_caption[key]) for key in sample]
        en_text_feature_sample = [data[1] for data in en_text_sample]
        
        return image_sample, en_text_feature_sample

    def clear_state(self):
        self.now_iter = 0
        self.now_epoch = 0
        self.now_loc = 0
    
    @property
    def get_state(self):
        return self.now_epoch, self.now_iter, self.now_loc
        
    @property
    def get_en_voc_size(self):
        return len(list(self.en_index2tok.keys()))
    
    @property
    def get_jp_voc_size(self):
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