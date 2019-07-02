import glob
import os
import pickle
import random
from collections import Counter

import numpy as np

from dataset.dataset_base import Dataset_Base
from utils.data_downloader import download_file_from_google_drive, downloder
from utils.progress import progress

script_path = os.path.dirname(__file__)
save_path = os.path.abspath(script_path + "/data")
if not os.path.exists(save_path):os.mkdir(save_path)
    
IMAGE_SIZE = 8189


class Oxford_102_flowers(Dataset_Base):
    def __init__(self, args, pre_train, shuffle=True):
        gpu_num = args.gpu_num
        batch_size = args.batch_size
        self.threshold = args.threshold
        data_path = os.path.dirname(__file__) + "/data"
        super(Oxford_102_flowers, self).__init__(gpu_num, batch_size, pre_train, shuffle)
        if not os.path.exists(data_path):os.mkdir(data_path)
        data_path = data_path + "/flower"
        if not os.path.exists(data_path):os.mkdir(data_path)
        self.data_path = data_path

        image_data = self.load_images()

        depth_data = self.load_depth(image_data)
            
        caption_data = self.load_captions()

        id_list = list(image_data.keys())
        self.train_image_data = {key:image_data[key] for key in id_list[:8000]}
        self.val_image_data = {key:image_data[key] for key in id_list[8000:]}

        self.train_depth_data = {key:depth_data[key] for key in id_list[:8000]}
        self.val_depth_data = {key:depth_data[key] for key in id_list[8000:]}

        self.train_caption_data = {key:caption_data[key] for key in id_list[:8000]}
        self.val_caption_data = {key:caption_data[key] for key in id_list[8000:]}

        self.train_id_list = list(self.train_image_data.keys())
        self.val_id_list = list(self.val_image_data.keys())

        if self.shuffle:
            np.random.shuffle(self.train_id_list)
            np.random.shuffle(self.val_id_list)

    def load_images(self):
        if os.path.exists(self.data_path + "/image_data.plk"):
            with open(self.data_path + "/image_data.plk", "rb") as fp:
                data = pickle.load(fp)
        else:
            file_name = "102flowers.tgz"
            url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
            if not os.path.exists(self.data_path + f"/{file_name}"):downloder(url, self.data_path + f"/{file_name}")
            
            if len(glob.glob(self.data_path + "/images/*.jpg")) != IMAGE_SIZE:
                print("Info:Extracting image data from tar file")
                import tarfile
                with tarfile.open(self.data_path + f"/{file_name}", 'r') as tar_fp:
                    tar_fp.extractall(self.data_path)
                os.rename(self.data_path + "/jpg", self.data_path + "/images")

            data = {}
            files = glob.glob(self.data_path + "/images/*.jpg")
            print("Info:load image data")
            for i, _path in enumerate(files):
                arr_id = int(_path.split("_")[-1].split(".")[0])
                arr_256 = self.path2array(_path, 256)
                arr_128 = self.path2array(_path, 128)
                arr_64 = self.path2array(_path, 64)
                data[arr_id] = {"x_256":arr_256,"x_128":arr_128,"x_64":arr_64}
                progress(i+1, IMAGE_SIZE)
            print("")

            with open(self.data_path + "/image_data.plk", "wb") as fp:
                pickle.dump(data, fp)
            
        return data

    def load_depth(self, image_data):
        if os.path.exists(self.data_path + "/depth_data.plk"):
            with open(self.data_path + "/depth_data.plk", "rb") as fp:
                depth_data = pickle.load(fp)
        else:
            print("Info:convert image to depth")
            depth_data = dict()
            key_list = list(image_data.keys())
            for i, key in enumerate(key_list):
                image = image_data[key]
                depth_128 = self.image2depth(image["x_256"])
                depth_64 = np.resize(depth_128,(64,64))
                depth_data[key] = {"x_128":depth_128,"x_64":depth_64}
                progress(i+1, len(key_list))
            
            with open(self.data_path + "/depth_data.plk", "wb") as fp:
                pickle.dump(depth_data, fp)
        
        return depth_data

    def load_captions(self):
        if os.path.exists(self.data_path + "/caption_data.pkl"):
            with open(self.data_path + "/caption_data.pkl", "rb") as fp:
                data = pickle.load(fp)
        else:
            file_name = "flowers_icml.tar.gz"
            drive_id = "0B0ywwgffWnLLMl9uOU91MV80cVU"
            if not os.path.exists(self.data_path + f"/{file_name}"):
                download_file_from_google_drive(drive_id, self.data_path + f"/{file_name}")

            if not os.path.exists(self.data_path + "/captions"):
                print("Info:Extracting caption data from tar file")
                import tarfile
                with tarfile.open(self.data_path + f"/{file_name}", 'r') as tar_fp:
                    tar_fp.extractall(self.data_path)
                    file_name = file_name.split(".")[0]
                    os.rename(self.data_path + f"/{file_name}", self.data_path + "/captions")
                    
            data = {}
            files = glob.glob(self.data_path + "/captions/*/*.t7")
            import torchfile
            for path in files:
                cap_id = int(path.split("_")[-1].split(".")[0])
                cap = torchfile.load(path)[b'txt']
                data[cap_id] = cap

            with open(self.data_path + "/caption_data.plk", "wb") as fp:
                pickle.dump(data, fp)

        return data