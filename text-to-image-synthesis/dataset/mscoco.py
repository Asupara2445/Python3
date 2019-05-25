import glob
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../../utils"))
from dataset.dataset_base import Dataset_Base
from utils.data_downloader import downloder
from utils.progress import progress

train_image_size = 82783
val_image_size = 40504

class MSCOCO(Dataset_Base):
    def __init__(self, image_size, gpu_num, batch_size, pre_train, shuffle=True):
        data_path = os.path.dirname(__file__) + "/data"
        super(MSCOCO, self).__init__(image_size, gpu_num, batch_size, pre_train, shuffle)
        if not os.path.exists(data_path):os.mkdir(data_path)
        data_path = data_path + "/mscoco"
        if not os.path.exists(data_path):os.mkdir(data_path)
        self.data_path = data_path
        
        self.train_caption_data, self.val_caption_data, self.embed_mat, self.index2tok, tok2index = self.load_captions()

        self.train_image_data = self.load_images()
        self.val_image_data = self.load_images(is_train=False)

        self.train_id_list = list(self.train_image_data.keys())
        self.val_id_list = list(self.val_image_data.keys())

        if self.shuffle:
            np.random.shuffle(self.train_data)
            np.random.shuffle(self.val_data)

    def load_captions(self):
        path = self.data_path + "/annotations"
        if not os.path.exists(path):os.mkdir(path)
        if os.path.exists(path + "/caption_data.pkl"):
            with open(path + "/caption_data.pkl", "rb") as fp:
                data = pickle.load(fp)
                return data["train_data"], data["val_data"], data["embed_mat"], data["index2tok"], data["tok2index"]
        else:
            url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
            file_name = url.split("/")[-1]
            if not os.path.exists(path + f"/{file_name}"):downloder(url, path + f"/{file_name}")

            import zipfile
            with zipfile.ZipFile(path + f"/{file_name}") as zip_fp:
                zip_fp.extractall(os.path.abspath(path + "/.."))

            print("Info:loading caption data")
            files = glob.glob(path + "/captions*.json")

            def load_caption(data_path):
                with open(data_path, "r", encoding="utf_8") as fp:
                    data = json.load(fp)
                    anns = data["annotations"]
                    data_dict = defaultdict(list)
                    for ann in anns:
                        data_dict[ann["image_id"]].append(ann["caption"])
                    return data_dict
            
            train = None;val = None
            for data_path in files:
                captions = load_caption(data_path)
                if "train" in data_path:train = captions
                elif "val" in data_path:val = captions
            
            train_data, val_data, embed_mat, index2tok, tok2index = self.make_vocaburaly(train, val, path)

            return train_data, val_data, embed_mat, index2tok, tok2index

    def load_images(self, is_train=True):
        if is_train:
            data_type = "train"
            data_len = train_image_size
        else:
            data_type = "val"
            data_len = val_image_size

        path = self.data_path + "/image"
        if not os.path.exists(path):os.mkdir(path)

        if os.path.exists(path + f"/{data_type}2014_array/{self.image_size[0]}"):
            files = glob.glob(path + f"/{data_type}2014_array/{self.image_size[0]}/*.npy")
            id_list = [int(path.split("_")[-1].split(".")[0]) for path in files]
            data = {id:path for id, path in zip(id_list,files)}

            return data
        else:
            url = f"http://images.cocodataset.org/zips/{data_type}2014.zip"
            file_name = url.split("/")[-1]
            if not os.path.exists(path + f"/{file_name}"):downloder(url, path + f"/{file_name}")

            if len(glob.glob(path + f"/{data_type}2014/*.jpg")) != data_len:
                print(f"Info:Extracting {data_type} image data from zip file")
                import zipfile
                with zipfile.ZipFile(path + f"/{file_name}") as zip_fp:
                    zip_fp.extractall(path)
            files = glob.glob(path + f"/{data_type}2014/*.jpg")

            path = path + f"/{data_type}2014_array"
            if not os.path.exists(path):os.mkdir(path)
            path = path + f"/{self.image_size[0]}"
            if not os.path.exists(path):
                os.mkdir(path)
                print(f"Info:Conveting {data_type} image data path to ndarray")
                for i, _path in enumerate(files):
                    file_name = _path.split("/")[-1].split(".")[0]
                    arr = self.path2array(_path)
                    np.save(path + f"/{file_name}.npy", arr)
                    progress(i+1, data_len)
                print("")

            files = glob.glob(path + "/*.npy")
            id_list = [int(path.split("_")[-1].split(".")[0]) for path in files]
            data = {id:path for id, path in zip(id_list,files)}

            return data

    def make_vocaburaly(self, train_data, val_data, save_path, threshold = 5):
        train_text = [caption for key in train_data.keys() for caption in train_data[key]]
        val_text = [caption for key in val_data.keys() for caption in val_data[key]]
        text_list = train_text + val_text
        text_counter = Counter([tok for text in text_list for tok in text.split(" ")])
        del text_counter[""]
        text_counter = {key:text_counter[key] for key in text_counter.keys() if text_counter[key] >= threshold}
        vocaburaly = list(text_counter.keys())
            
        embed_mat, index2tok, tok2index = self.make_embed_mat(vocaburaly)
            
        ignore_toks = vocaburaly + [key for key in text_counter.keys() if text_counter[key] < threshold]

        train_data = {key:[np.array([0] + [tok2index.get(tok, 2) for tok in caption.split(" ") if tok != ""] + [1]) for caption in train_data[key]] for key in train_data.keys()}
        val_data = {key:[np.array([0] + [tok2index.get(tok, 2) for tok in caption.split(" ") if tok != ""] + [1]) for caption in val_data[key]] for key in val_data.keys()}

        with open(save_path + "/caption_data.pkl", "wb") as fp:
            data = {"train_data":train_data,"val_data":val_data,"embed_mat":embed_mat,"index2tok":index2tok,"tok2index":tok2index}
            pickle.dump(data, fp)
        
        return train_data, val_data, embed_mat, index2tok, tok2index