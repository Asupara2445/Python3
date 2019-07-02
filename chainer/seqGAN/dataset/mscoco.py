import glob
import os
import json
import pickle
import re
import string
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import numpy as np

from utils.data_downloader import downloder, download_file_from_google_drive
from utils.progress import progress


def load_english_captions(sent_len, data_path, threshpld):
    if os.path.exists(data_path + "/english_captions.pkl"):
        with open(data_path + "/english_captions.pkl", "rb") as fp:
            data = pickle.load(fp)
            return data["train_data"], data["val_data"], data["index2tok"]
    else:
        url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
        file_name = "annotations_trainval2014.zip"
        if not os.path.exists(data_path + f"/{file_name}"):
            downloder(url, data_path + f"/{file_name}")
        
        annotations_dir = data_path + "/annotations"
        if  not os.path.exists(annotations_dir):os.mkdir(annotations_dir)
        if len(os.listdir(annotations_dir)) != 6:
            print("Info:extracting zip file")
            import zipfile
            with zipfile.ZipFile(data_path + f"/{file_name}", 'r') as zip_fp:
                zip_fp.extractall(data_path)
        
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
        for _path in files:
            captions = load_caption(_path)
            if "train" in _path:train = captions
            elif "val" in _path:val = captions
        
        train_data, val_data, index2tok, tok2index = make_vocaburaly(train, val, sent_len, threshpld)

        with open(data_path + "/english_captions.pkl", "wb") as fp:
            data = {"train_data":train_data, "val_data":val_data, "index2tok":index2tok, "tok2index":tok2index}
            pickle.dump(data, fp)
        
        return train_data, val_data, index2tok

def load_japanese_captions(sent_len, data_path, threshpld):
    if os.path.exists(data_path + "/japanese_caption_data.pkl"):
        with open(data_path + "/japanese_caption_data.pkl", "rb") as fp:
            data = pickle.load(fp)
            return data["train_data"], data["val_data"], data["index2tok"]
    else:
        url = "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.2.tar.gz"
        file_name = "stair_captions_v1.2.tar.gz"
        if not os.path.exists(data_path + f"/{file_name}"):downloder(url, data_path + f"/{file_name}")

        path = data_path + "/stair_captions_v1.2"
        if not os.path.exists(path):
            os.mkdir(path)
            print("Info:extracting caption data")
            import tarfile
            with tarfile.open(data_path + f"/{file_name}", "r") as tar_fp:
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
        for _path in files:
            captions = load_caption(_path)
            if "train" in _path:train = captions
            elif "val" in _path:val = captions
        
        train_data, val_data, index2tok, tok2index = make_vocaburaly(train, val, sent_len, threshpld)

        with open(data_path + "/japanese_caption_data.pkl", "wb") as fp:
            data = {"train_data":train_data, "val_data":val_data, "index2tok":index2tok, "tok2index":tok2index}
            pickle.dump(data, fp)

        return train_data, val_data, index2tok

def make_vocaburaly(train_data, val_data, sent_len, threshpld):
    train_text = [caption for key in train_data for caption in train_data[key]]
    val_text = [caption for key in val_data for caption in val_data[key]]
    captions = train_text + val_text

    word_counts = defaultdict(float)
    for sent in captions:
        for word in sent:
            word_counts[word] += 1

    vocab = [w for w in word_counts if word_counts[w] >= threshpld]

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