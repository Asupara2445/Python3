import os
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

class Dataset_Base():
    def __init__(self, gpu_num, batch_size, pre_train, shuffle):
        self._batch_size = batch_size
        self.gpu_num = gpu_num
        self.pre_train = pre_train
        self.shuffle = shuffle
        self.image_keys = ["x_64", "x_128", "x_256"]

        self.now_iter = 0
        self.now_epoch = 1
        self.now_loc = 0

        self.train_id_list = None;self.val_id_list = None
        self.train_image_data = None;self.train_caption_data = None;self.train_depth_data = None
        self.val_image_data = None;self.val_caption_data = None;self.val_depth_data = None
    
    def next(self):
        batch = self.train_id_list[self.now_loc:self.now_loc+self._batch_size]
        if len(batch) < self._batch_size:
            self.now_loc = self._batch_size-len(batch)
            if self.shuffle: np.random.shuffle(self.train_id_list)
            add_batch = self.train_id_list[:self._batch_size-len(batch)]
            batch.extend(add_batch)
            self.now_epoch += 1
        self.now_loc += self._batch_size

        image_batch = [self.train_image_data[key] for key in batch]
        depth_batch = [self.train_depth_data[key] for key in batch]
        text_batch =  [random.choice(self.train_caption_data[key]) for key in batch]

        wrong_batch = self.train_id_list[self.now_loc:self.now_loc+self._batch_size]
        if len(wrong_batch) < self._batch_size:
            add_batch = self.train_id_list[:self._batch_size-len(wrong_batch)]
            wrong_batch.extend(add_batch)

        wrong_image_batch = [self.train_image_data[key] for key in wrong_batch]
        wrong_depth_batch = [self.train_depth_data[key] for key in wrong_batch]
        wrong_text_batch = [random.choice(self.train_caption_data[key]) for key in wrong_batch]

        self.now_iter += 1

        return image_batch, depth_batch, np.array(text_batch, dtype="float32"), wrong_image_batch, wrong_depth_batch, np.array(wrong_text_batch, dtype="float32")

    def path2array(self, path, image_size):
        img = cv2.imread(path)
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2,0,1).astype("float32") / 255 - 0.5
        return img

    def image2depth(self, img):
        try:
            sess = getattr(self, "sess")
            depth_net = getattr(self, "depth_net")
            input_node = getattr(self, "input_node")
        except:
            import sys
            import tensorflow as tf
            os.environ[ 'TF_CPP_MIN_LOG_LEVEL'] = '0'
            sys.path.append(os.path.abspath(__file__ + "/../models"))
            from models.fcrn import ResNet50UpProj
            model_params_path = os.path.abspath(__file__ + "/../data/NYU_ResNet-UpProj.npy")

            if not os.path.exists(model_params_path):
                url = "http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy"
                downloder(url, model_params_path)

            self.input_node = tf.placeholder(tf.float32, shape=(None, 256, 256, 3))
            self.depth_net = ResNet50UpProj({'data': self.input_node}, 1, 1, False)

            self.sess = tf.Session()
            print('Loading the model')
            self.depth_net.load(model_params_path, self.sess)

            sess = getattr(self, "sess")
            depth_net = getattr(self, "depth_net")
            input_node = getattr(self, "input_node")
                
        def normalization(arr):
            _min, _max = np.min(arr), np.max(arr)
            arr = (arr - _min) / (_max - _min)
            return arr

        img = img.transpose((1,2,0))
        img = np.expand_dims(np.asarray(img), axis = 0)
        pred = np.asarray(sess.run(depth_net.get_output(), feed_dict={input_node: img}))[0,:,:,0]

        return normalization(pred)

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