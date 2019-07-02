import glob
import os
import json
import pickle
import re
import string
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import numpy as np

from dataset.dataset_base import Dataset_Base
from utils.data_downloader import downloder, download_file_from_google_drive
from utils.progress import progress

train_image_size = 82783
val_image_size = 40504

class MSCOCO(Dataset_Base):
    def __init__(self, conf, shuffle=True):
        gpu_num = conf.gpu_num
        use_lang = conf.use_lang
        batch_size = conf.batch_size
        data_path = os.path.dirname(__file__) + "/data"
        super(MSCOCO, self).__init__(gpu_num, batch_size, use_lang, shuffle)
        if not os.path.exists(data_path):os.mkdir(data_path)
        self.data_path = data_path
        self.cut_freq = conf.cut_freq
        
        self.train_en_caption, self.val_en_caption, self.en_index2tok  = self.load_english_captions(conf.seq_len_en)
        
        self.train_jp_caption, self.val_jp_caption, self.jp_index2tok  = self.load_japanese_captions(conf.seq_len_jp)

        self.train_image_data = self.load_images()
        self.val_image_data = self.load_images(is_train=False)

        self.train_id_list = list(self.train_en_caption.keys())
        self.val_id_list = list(self.val_en_caption.keys())

        #self.train_id_list = self.train_id_list[:2000]
        #self.val_id_list = self.val_id_list[:400]

        if self.shuffle:
            np.random.shuffle(self.train_id_list)
            np.random.shuffle(self.val_id_list)

    def load_images(self, is_train=True):
        if is_train:
            data_type = "train"
            data_len = train_image_size
        else:
            data_type = "val"
            data_len = val_image_size

        if len(glob.glob(self.data_path + f"/{data_type}2014_array/*.npy")) == data_len:
            files = glob.glob(self.data_path + f"/{data_type}2014_array/*.npy")
            id_list = [int(path.split("_")[-1].split(".")[0]) for path in files]
            data = {id:path for id, path in zip(id_list,files)}

            return data
        else:
            url = f"http://images.cocodataset.org/zips/{data_type}2014.zip"
            file_name = url.split("/")[-1]
            if not os.path.exists(self.data_path + f"/{file_name}"):downloder(url, self.data_path + f"/{file_name}")

            path = self.data_path + f"/{data_type}2014_array"
            if len(glob.glob(self.data_path + f"/{data_type}2014/*.jpg")) != data_len:
                print(f"Info:Extract {data_type} images from zip file and convert images to ndarray")
                if not os.path.exists(path):os.mkdir(path)
                import zipfile
                with zipfile.ZipFile(self.data_path + f"/{file_name}") as zip_fp:
                    file_count = len(zip_fp.filelist)
                    for i, item in enumerate(zip_fp.filelist):
                        file_name = item.filename.split("/")[-1].split(".")[0]
                        zip_fp.extract(item, self.data_path)
                        if file_name != "":
                            arr = self.path2array(self.data_path + f"/{data_type}2014/{file_name}.jpg")
                            np.save(path + f"/{file_name}", arr)
                        progress(i+1, file_count)
                    print("")

            files = glob.glob(path + "/*.npy")
            id_list = [int(path.split("_")[-1].split(".")[0]) for path in files]
            data = {id:path for id, path in zip(id_list,files)}

            return data

    def load_english_captions(self, sent_len):
        if os.path.exists(self.data_path + "/english_captions.pkl"):
            with open(self.data_path + "/english_captions.pkl", "rb") as fp:
                data = pickle.load(fp)
                return data["train_data"], data["val_data"], data["index2tok"]
        else:
            url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
            file_name = "annotations_trainval2014.zip"
            if not os.path.exists(self.data_path + f"/{file_name}"):
                downloder(url, self.data_path + f"/{file_name}")
            
            annotations_dir = self.data_path + "/annotations"
            if  not os.path.exists(annotations_dir):os.mkdir(annotations_dir)
            if len(os.listdir(annotations_dir)) != 6:
                print("Info:extracting zip file")
                import zipfile
                with zipfile.ZipFile(self.data_path + f"/{file_name}", 'r') as zip_fp:
                    zip_fp.extractall(self.data_path)
            
            print("Info:loading english captin data")
            def load_caption(data_path):
                def preprocess_caption(line):
                        line = line.replace("\ufffd\ufffd", " ")
                        tokenizer = RegexpTokenizer(r"\w+")
                        tokens = tokenizer.tokenize(line.lower())
                        return tokens

                with open(data_path, "r", encoding="utf_8") as fp:
                    data = json.load(fp)
                    anns = data["annotations"]
                    data_dict = defaultdict(list)
                    for ann in anns:
                        data_dict[ann["image_id"]].append(preprocess_caption(ann["caption"]))
                    return data_dict
            
            files = glob.glob(annotations_dir + "/captions_*2014.json")
            train = None;val = None
            for data_path in files:
                captions = load_caption(data_path)
                if "train" in data_path:train = captions
                elif "val" in data_path:val = captions
            
            train_data, val_data, index2tok, tok2index = self.make_vocaburaly(train, val, sent_len)

            with open(self.data_path + "/english_captions.pkl", "wb") as fp:
                data = {"train_data":train_data, "val_data":val_data, "index2tok":index2tok, "tok2index":tok2index}
                pickle.dump(data, fp)
            
            return train_data, val_data, index2tok

    def load_japanese_captions(self, sent_len):
        if os.path.exists(self.data_path + "/japanese_caption_data.pkl"):
            with open(self.data_path + "/japanese_caption_data.pkl", "rb") as fp:
                data = pickle.load(fp)
                return data["train_data"], data["val_data"], data["index2tok"]
        else:
            url = "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.2.tar.gz"
            file_name = "stair_captions_v1.2.tar.gz"
            if not os.path.exists(self.data_path + f"/{file_name}"):downloder(url, self.data_path + f"/{file_name}")

            path = self.data_path + "/stair_captions_v1.2"
            if not os.path.exists(path):
                os.mkdir(path)
                print("Info:extracting caption data")
                import tarfile
                with tarfile.open(self.data_path + f"/{file_name}", "r") as tar_fp:
                        tar_fp.extractall(path)
            files = glob.glob(path + "/*_tokenized.json")

            print(f"Info:loading japanese caption data")
            def load_caption(data_path):
                def preprocess_caption(line):
                        prep_line = re.sub("[%s]" % re.escape(string.punctuation), " ", line.rstrip())
                        prep_line = prep_line.replace("-", " ").replace("\n", "")
                        return prep_line.lower().split(" ")

                with open(data_path, "r", encoding="utf_8") as fp:
                    data = json.load(fp)
                    anns = data["annotations"]
                    data_dict = defaultdict(list)
                    for ann in anns:
                        data_dict[ann["image_id"]].append(preprocess_caption(ann["tokenized_caption"]))
                    return data_dict
            
            train = None;val = None
            for data_path in files:
                captions = load_caption(data_path)
                if "train" in data_path:train = captions
                elif "val" in data_path:val = captions
            
            train_data, val_data, index2tok, tok2index = self.make_vocaburaly(train, val, sent_len)

            with open(self.data_path + "/japanese_caption_data.pkl", "wb") as fp:
                data = {"train_data":train_data, "val_data":val_data, "index2tok":index2tok, "tok2index":tok2index}
                pickle.dump(data, fp)

            return train_data, val_data, index2tok

    def make_vocaburaly(self, train_data, val_data, sent_len):
        train_text = [caption for key in train_data for caption in train_data[key]]
        val_text = [caption for key in val_data for caption in val_data[key]]
        captions = train_text + val_text

        word_counts = defaultdict(float)
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= self.cut_freq]

        index2tok = {}
        index2tok[0] = '<end>'
        index2tok[1] = '<unk>'
        tok2index = {}
        tok2index['<end>'] = 0
        tok2index['<unk>'] = 1
        ix = 2
        for w in vocab:
            tok2index[w] = ix
            index2tok[ix] = w
            ix += 1

        def sequential_pad(arr, limit):
            pad_arr = np.pad(arr, (0, limit-len(arr)), "constant", constant_values=0)
            return pad_arr

        train_data = {key:[sequential_pad([tok2index.get(tok, 1) for tok in caption[:sent_len] if tok != ""], sent_len) for caption in train_data[key]] for key in train_data}
        val_data = {key:[sequential_pad([tok2index.get(tok, 1) for tok in caption[:sent_len] if tok != ""], sent_len) for caption in val_data[key]] for key in val_data}
        
        return train_data, val_data, index2tok, tok2index