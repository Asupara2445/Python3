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

IMAGE_SIZE = 11788


class CUB(Dataset_Base):
    def __init__(self, args, pre_train, shuffle=True):
        image_size = args.image_size
        gpu_num = args.gpu_num
        batch_size = args.batch_size
        data_path = os.path.dirname(__file__) + "/data"
        super(CUB, self).__init__(image_size, gpu_num, batch_size, pre_train, shuffle)
        if not os.path.exists(data_path):os.mkdir(data_path)
        data_path = data_path + "/bird"
        if not os.path.exists(data_path):os.mkdir(data_path)
        self.data_path = data_path

        image_data_64 = self.load_images(64)
        image_data_128 = self.load_images(128)
        image_data_256 = self.load_images(256)
        
        caption_data = self.load_captions()
    
        id_list = list(image_data_64.keys())
        self.train_image_data = {key:{"x_64":image_data_64[key], "x_128":image_data_128[key], "x_256":image_data_256[key]} for key in id_list[:10000]}
        self.val_image_data = {key:{"x_64":image_data_64[key], "x_128":image_data_128[key], "x_256":image_data_256[key]} for key in id_list[10000:]}

        self.train_caption_data = {key:caption_data[key] for key in id_list[:10000]}
        self.val_caption_data = {key:caption_data[key] for key in id_list[10000:]}

        self.train_id_list = list(self.train_image_data.keys())
        self.val_id_list = list(self.val_image_data.keys())

        if self.shuffle:
            np.random.shuffle(self.train_id_list)
            np.random.shuffle(self.val_id_list)


    def load_images(self, img_size):
        if os.path.exists(self.data_path + f"/image_data_{img_size}.plk"):
            with open(self.data_path + f"/image_data_{img_size}.plk", "rb") as fp:
                data = pickle.load(fp)
        else:
            file_name = "CUB_200_2011.tgz"
            url = "http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
            if not os.path.exists(self.data_path + f"/{file_name}"):downloder(url, self.data_path + f"/{file_name}")
            
            if len(glob.glob(self.data_path + "/images/*.jpg")) != IMAGE_SIZE:
                print("Info:Extracting image data from tar file")
                import tarfile, shutil
                with tarfile.open(self.data_path + f"/{file_name}", 'r') as tar_fp:
                    tar_fp.extractall(self.data_path)
                    shutil.move(self.data_path + "/CUB_200_2011/images", self.data_path)
            
            data = {}
            files = glob.glob(self.data_path + "/images/*/*.jpg")
            print(f"Info:load {img_size}x{img_size} image data")
            for i, _path in enumerate(files):
                arr_id = int(_path.split("_")[-1].split(".")[0])
                arr = self.path2array(_path, img_size)
                data[arr_id] = arr
                progress(i+1, IMAGE_SIZE)
            print("")

            with open(self.data_path + f"/image_data_{img_size}.plk", "wb") as fp:
                pickle.dump(data, fp)
            
        return data
    

    def load_captions(self):
        if os.path.exists(self.data_path + "/caption_data.pkl"):
            with open(self.data_path + "/caption_data.pkl", "rb") as fp:
                data = pickle.load(fp)
        else:
            file_name = "cub_icml.tar.gz"
            drive_id = "0B0ywwgffWnLLLUc2WHYzM0Q2eWc"
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