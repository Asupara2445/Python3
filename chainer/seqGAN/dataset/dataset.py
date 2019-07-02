import os
import random

import chainer
import numpy as np

from dataset.mscoco import load_english_captions, load_japanese_captions


def get_Dataset(conf):
    data_path = os.path.join(os.path.dirname(__file__) + "/data")
    threshold = conf.cut_count
    if not os.path.exists(data_path):os.mkdir(data_path)
    if conf.use_lang == "en":
        seq_len = conf.seq_len_en
        train_caption, val_caption, index2tok  = load_english_captions(seq_len, data_path, threshold)
    elif  conf.use_lang == "jp":
        seq_len = conf.seq_len_jp
        train_caption, val_caption, index2tok  = load_japanese_captions(seq_len, data_path, threshold)

    return Dataset(train_caption), Dataset(val_caption), index2tok, seq_len


class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, captions, shuffle=True):
        data_path = os.path.join(os.path.dirname(__file__) + "/data")
        self.captions = captions

        self.id_list = list(self.captions.keys())

        if shuffle:
            np.random.shuffle(self.id_list)
    
    def __len__(self):
        return len(self.id_list)

    def get_example(self, i):
        key = self.id_list[i]

        caption = random.choice(self.captions[key])

        return np.array(caption)
