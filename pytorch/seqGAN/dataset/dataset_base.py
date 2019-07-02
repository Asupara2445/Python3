import os
import random

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
        self.train_en_caption = None; self.train_jp_caption = None
        self.val_en_caption = None;self.val_jp_caption = None
    
    def get_data(self, is_val=False, is_pretrain=False):
        if not is_val:use_data = "train"
        else:use_data = "val"

        id_list = getattr(self, f"{use_data}_id_list")
        if self.use_lang == "jp":
            caption = getattr(self, f"{use_data}_jp_caption")
        elif  self.use_lang == "en":
            caption = getattr(self, f"{use_data}_en_caption")

        self.now_loc = 0
        self.now_epoch += 1

        for i in range(0, len(id_list), self.batch_size):
            batch = id_list[i:i+self._batch_size]
            self.now_loc += self._batch_size

            text = [random.choice(caption[key]) for key in batch]

            wrong = id_list[self.now_loc:self.now_loc+self._batch_size]
            wrong_text = [random.choice(caption[key]) for key in wrong]

            self.now_iter += 1

            yield text, wrong_text

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