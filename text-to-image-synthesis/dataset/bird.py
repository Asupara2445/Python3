import glob
import json
import os
import pickle
import random
import re
import string
import sys
from collections import Counter

import numpy as np

from dataset.dataset_base import Dataset_Base
from utils.data_downloader import download_file_from_google_drive, downloder
from utils.progress import progress

script_path = os.path.dirname(__file__)
save_path = os.path.abspath(script_path + "/data")
if not os.path.exists(save_path):os.mkdir(save_path)

image_size = 11788

class CUB(Dataset_Base):
    def __init__(self, image_size, gpu_num, batch_size, shuffle=True):
        data_path = os.path.dirname(__file__) + "/data"
        super(CUB, self).__init__(image_size, gpu_num, batch_size, shuffle)
        if not os.path.exists(data_path):os.mkdir(data_path)
        data_path = data_path + "/bird"
        if not os.path.exists(data_path):os.mkdir(data_path)
        self.data_path = data_path

        image_data = self.load_images()
        
        caption_data, self.embed_mat, self.index2tok, tok2index = self.load_captions()
    
        id_list = list(image_data.keys())
        self.train_image_data = {key:image_data[key] for key in id_list[:10000]}
        self.val_image_data = {key:image_data[key] for key in id_list[10000:]}

        self.train_caption_data = {key:caption_data[key] for key in id_list[:10000]}
        self.val_caption_data = {key:caption_data[key] for key in id_list[10000:]}

        self.train_id_list = list(self.train_image_data.keys())
        self.val_id_list = list(self.val_image_data.keys())

        if self.shuffle:
            np.random.shuffle(self.train_id_list)
            np.random.shuffle(self.val_id_list)

    def load_images(self):
        files_cont = len(glob.glob(self.data_path + f"/array/{self.image_size[0]}/*.npy"))
        if files_cont == image_size:
            files = glob.glob(self.data_path + f"/array/{self.image_size[0]}/*.npy")
            id_list = [int(path.split("_")[-1].split(".")[0]) for path in files]
            data = {id:path for id, path in zip(id_list,files)}
        else:
            url = "http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
            title = "CUB_200_2011.tgz"
            if not os.path.exists(self.data_path + f"/{title}"):downloder(url, self.data_path + f"/{title}")
            
            if len(glob.glob(self.data_path + "/image/*/*.jpg")) != image_size:
                print("Info:Extracting image data from tar file")
                import tarfile, shutil
                with tarfile.open(self.data_path + f"/{title}", 'r') as tar_fp:
                    tar_fp.extractall(self.data_path)
                shutil.move(self.data_path + "/CUB_200_2011/images", self.data_path)
                os.rename(self.data_path + "/images", self.data_path + "/image")

            files = glob.glob(self.data_path + "/image/*/*.jpg")
            path = self.data_path + "/array"
            if not os.path.exists(path):os.mkdir(path)
            path = path + f"/{self.image_size[0]}"
            if not os.path.exists(path):
                os.mkdir(path)
                print("Info:Conveting image data path to ndarray")
                for i, _path in enumerate(files):
                    file_name = _path.split("/")[-1].split(".")[0]
                    arr = self.path2array(_path)
                    np.save(path + f"/{file_name}.npy", arr)
                    progress(i+1, image_size)
                print("")

            files = glob.glob(path + "/*.npy")
            id_list = [int(path.split("_")[-1].split(".")[0]) for path in files]
            data = {id:path for id, path in zip(id_list,files)}

        return data
    
    def load_captions(self):
        path = self.data_path + "/captions"
        if not os.path.exists(path):os.mkdir(path)

        if os.path.exists(path + "/caption_data.pkl"):
            with open(path + "/caption_data.pkl", "rb") as fp:
                data = pickle.load(fp)

                return data["data"], data["embed_mat"], data["index2tok"], data["tok2index"]
        else:
            file_name = "cvpr2016_cub.tar.gz"
            if not os.path.exists(self.data_path + f"/{file_name}"):
                drive_id = "0B0ywwgffWnLLZW9uVHNjb2JmNlE"
                download_file_from_google_drive(drive_id, self.data_path + f"/{file_name}")

            if len(glob.glob(path + "/*")) != 15:
                print("Info:Extracting caption data from tar file")
                import tarfile
                with tarfile.open(self.data_path + f"/{file_name}", 'r') as tar_fp:
                    tar_fp.extractall(path)

            def preprocess_caption(line):
                prep_line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line.rstrip())
                prep_line = prep_line.replace('-', ' ')
                return prep_line

            data = dict()
            path = path + "/text_c10"
            dir_list = [os.path.join(path,o) for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]
            for i, _dir in enumerate(dir_list):
                files = glob.glob(_dir + "/*.txt")
                for j, _path in enumerate(files):
                    key = int(_path.split("_")[-1].split(".")[0])
                    with open(_path, "r", encoding="utf_8") as fp:
                        captions = [preprocess_caption(line) for line in fp.readlines()]
                    data[key] = captions

            data, embed_mat, index2tok, tok2index = self.make_vocaburaly(data, self.data_path + "/captions")

            return data, embed_mat, index2tok, tok2index

    def make_vocaburaly(self, data, save_path, threshold = 5):
        text_list = [caption for key in data.keys() for caption in data[key]]
        text_counter = Counter([tok for text in text_list for tok in text.split(" ")])
        del text_counter[""]
        text_counter = {key:text_counter[key] for key in text_counter.keys() if text_counter[key] >= threshold}
        vocaburaly = list(text_counter.keys())
            
        embed_mat, index2tok, tok2index = self.make_embed_mat(vocaburaly)
            
        ignore_toks = vocaburaly + [key for key in text_counter.keys() if text_counter[key] < threshold]

        data = {key:[np.array([0] + [tok2index.get(tok, 2) for tok in caption.split(" ") if tok != ""] + [1]) for caption in data[key]] for key in data.keys()}

        with open(save_path + "/caption_data.pkl", "wb") as fp:
            data = {"data":data,"embed_mat":embed_mat,"index2tok":index2tok,"tok2index":tok2index}
            pickle.dump(data, fp)
        
        return data["data"], embed_mat, index2tok, tok2index