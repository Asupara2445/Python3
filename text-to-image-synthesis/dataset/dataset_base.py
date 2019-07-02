import os
import glob
import json
import pickle
import random

import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

from updater import Updater
from utils.progress import print_progress
from utils.data_downloader import downloder

class Dataset_Base():
    def __init__(self, image_size, gpu_num, batch_size, shuffle):
        self._batch_size = batch_size
        self.image_size = image_size
        self.gpu_num = gpu_num
        self.shuffle = shuffle

        self.now_iter = 0
        self.now_epoch = 0
        self.now_loc = 0

        self.train_id_list = None;self.val_id_list = None
        self.train_image_data = None;self.train_caption_data = None
        self.val_image_data = None;self.val_caption_data = None
    
    def next(self):
        batch = self.train_id_list[self.now_loc:self.now_loc+self._batch_size]
        if len(batch) < self._batch_size:
            self.now_loc = self._batch_size-len(batch)
            if self.shuffle: np.random.shuffle(self.train_id_list)
            add_batch = self.train_id_list[:self._batch_size-len(batch)]
            batch.extend(add_batch)
            self.now_epoch += 1
        self.now_loc += self._batch_size

        image_batch = [np.load(self.train_image_data[key]) for key in batch]
        text_batch =  [random.choice(self.train_caption_data[key]) for key in batch]

        wrong_batch = self.train_id_list[self.now_loc:self.now_loc+self._batch_size]
        if len(wrong_batch) < self._batch_size:
            add_batch = self.train_id_list[:self._batch_size-len(wrong_batch)]
            wrong_batch.extend(add_batch)

        wrong_image_batch = [np.load(self.train_image_data[key]) for key in wrong_batch]
        wrong_text_batch = [random.choice(self.train_caption_data[key]) for key in wrong_batch]

        self.now_iter += 1

        return np.array(image_batch), text_batch, np.array(wrong_image_batch), wrong_text_batch

    def path2array(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, self.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2,0,1).astype("float32") / 255 - 0.5
        return img
    
    def make_embed_mat(self, vocaburaly):
        embed_mat_path = os.path.abspath(self.data_path + "/../crawl-300d-2M.vec")
        if not os.path.exists(embed_mat_path):
            url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
            if not os.path.exists(embed_mat_path + ".zip"):downloder(url, embed_mat_path + ".zip")
            import zipfile
            with zipfile.ZipFile(embed_mat_path + ".zip") as zip_fp:
                zip_fp.extractall(os.path.abspath(self.data_path + "/../"))
            
        with open(embed_mat_path, "r", encoding="utf_8") as fp:
            n, d = map(int, fp.readline().split())
            embed_data = {}
            for line in fp:
                tokens = line.rstrip().split(' ')
                if tokens[0] in vocaburaly:
                    embed_data[tokens[0]] = [float(v) for v in tokens[1:]]
                    del vocaburaly[vocaburaly.index(tokens[0])]
            index2tok = {i+3:key for i, key in enumerate(embed_data.keys())}
            tok2index = {key:i+3 for i, key in enumerate(embed_data.keys())}
            embed_mat = np.array([np.random.normal(size=d),np.random.normal(size=d),np.random.normal(size=d)] + [embed_data[index2tok[i+3]] for i in range(len(index2tok.keys()))])
            index2tok[0] = "<S>";index2tok[1] = "</S>";index2tok[2] = "<UNK>"
            tok2index["<S>"] = 0;tok2index["</S>"] = 1;tok2index["<UNK>"] = 2

        return embed_mat, index2tok, tok2index

    def sampling(self, size=10):
        sample =  random.sample(self.train_id_list, size)
        
        image_sample = [np.load(self.train_image_data[key]) for key in sample]
        text_sample =  [random.choice(self.train_caption_data[key]) for key in sample]
                
        return np.array(image_sample), text_sample

    def clear_state(self):
        self.now_iter = 0
        self.now_epoch = 0
        self.now_loc = 0
    
    @property
    def get_state(self):
        return self.now_epoch, self.now_iter, self.now_loc
        
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
        return len(list(self.val_image_data.keys()))