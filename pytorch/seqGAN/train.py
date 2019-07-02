import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from dataset.mscoco import MSCOCO
from models.generator import GENERATOR
from models.discriminator import DISCRIMINATOR
from utils.configure import get_args
from utils.progress import print_progress, progress
from updater import Updater

script_path = os.path.dirname(__file__)


def toGPU(batch, gpu_num):
    class Data():
        def __init__(self):pass
    
    data = Data

    if gpu_num < 0:
        text = torch.tensor(batch[0])
        #wrong_text = torch.tensor(batch[1])

    else:
        text = torch.tensor(batch[0]).cuda()
        #wrong_text = torch.tensor(batch[1]).cuda()

    setattr(data, "text", text)
    #setattr(data, "wrong_text", wrong_text)
    
    return data

def build_models(conf, n_voc, seq_len):
    netG = GENERATOR(n_voc, conf.emb_dim, conf.hid_dim, seq_len, conf.gpu_num)
    netD = DISCRIMINATOR(n_voc, conf.emb_dim, conf.hid_dim, seq_len, conf.gpu_num)

    if conf.gpu_num > 0:
        netG.cuda()
        netD.cuda()

    if conf.gpu_num >= 2:
        netG = nn.DataParallel(netG, device_ids=[i for i in range(conf.gpu_num)])
        netD = nn.DataParallel(netD, device_ids=[i for i in range(conf.gpu_num)])

    return netG, netD

def build_optimizer(netG, netD, lr, beta1, beta2):

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))

    return optimizerG, optimizerD

def sample_generate(netG, sample_size, index2tok, save_path):
    sample = netG(sample_size)
    sample = sample.data.cpu().numpy()
    sample = [" ".join([index2tok[idx] for idx in text]) for text in sample]

    with open(save_path, "w", encoding="utf_8") as fp:
        sample = "\n".join(sample)
        fp.write(sample)


def main():
    conf = get_args()
    dataset = MSCOCO(conf)
    VOC_SIZE = dataset.jp_voc_size if conf.use_lang == "jp" else  dataset.en_voc_size
    SAMPLE_SIZE = conf.sample_size // conf.gpu_num if conf.gpu_num > 1 else conf.sample_size
    SEQ_LEN = conf.seq_len_jp if conf.use_lang == "jp" else   conf.seq_len_en
    index2tok = dataset.jp_index2tok if conf.use_lang == "jp" else  dataset.en_index2tok

    if not conf.silent:
        save_path = os.path.abspath(script_path + conf.save_path)
        if not os.path.exists(save_path):os.mkdir(save_path)
        save_path = os.path.abspath(save_path + f"/{conf.use_lang}")
        if not os.path.exists(save_path):os.mkdir(save_path)
        preview_path = os.path.abspath(save_path + "/preview")
        if not os.path.exists(preview_path):os.mkdir(preview_path)

    netG, netD = build_models(conf, VOC_SIZE, SEQ_LEN)
    optimizerG, optimizerD = build_optimizer(netG, netD, conf.adam_lr, conf.adam_beta1, conf.adam_beta2)
    pprog = print_progress(conf.pre_gen_max_epoch, conf.batch_size, dataset.train_data_len)

    updater = Updater(netG, netD, optimizerG, optimizerD, conf)
    """
    print("==========================================")
    print("Info:start genarator pre train")
    pre_gen_loss_hist = np.zeros((1, conf.pre_gen_max_epoch), dtype="float32")
    for i in range(conf.pre_gen_max_epoch):
        count = 0
        total_loss = np.array([0.], dtype="float32")
        start = time.time()
        for data in dataset.get_data():
            break
            data = toGPU(data, conf.gpu_num)
            loss = updater.update_pre_gen(data)

            total_loss += loss.data.cpu().numpy()

            count += 1
            if dataset.now_iter % conf.display_interval == 0:
                elapsed = time.time() - start
                pprog(elapsed, dataset.get_state)
                start = time.time()

        pre_gen_loss_hist[0,i] = total_loss / count

    if not conf.silent:
        sample_generate(netG, SAMPLE_SIZE, index2tok, preview_path + f"/sample_text_pretrain.txt")
        np.save(save_path + "/pre_gen_loss_hist", pre_gen_loss_hist)
        torch.save(netG.state_dict(), save_path + "/pretrain_gen_params")
    print("\n\n\n\n==========================================")


    print("==========================================")
    print("Info:start discriminator pre train")
    dataset.clear_state()
    pprog.max_iter = conf.pre_dis_max_epoch
    pre_dis_hist = np.zeros((2, conf.pre_dis_max_epoch), dtype="float32")
    for i in range(conf.pre_dis_max_epoch):
        count = 0
        total_loss = np.array([0.], dtype="float32")
        total_acc = np.array([0.], dtype="float32")
        start = time.time()
        for data in dataset.get_data():
            data = toGPU(data, conf.gpu_num)
            loss, acc = updater.update_dis(data)

            total_loss += loss.data.cpu().numpy()
            total_acc += acc.data.cpu().numpy()

            count += 1
            if dataset.now_iter % conf.display_interval == 0:
                elapsed = time.time() - start
                pprog(elapsed, dataset.get_state)
                start = time.time()

        pre_dis_hist[0,i] = total_loss / count
        pre_dis_hist[1,i] = total_acc / count

    if not conf.silent:
        np.save(save_path + "/pre_dis_hist", pre_dis_hist)
        torch.save(netD.state_dict(), save_path + "/pretrain_dis_params")
    print("\n\n\n\n==========================================")
    """
            

    print("==========================================")
    print("Info:start main train")
    dataset.clear_state()
    pprog.max_iter = conf.max_epoch
    loss_hist = np.zeros((3, conf.max_epoch), dtype="float32")
    for i in range(conf.max_epoch):
        count = 0
        total_g_loss = np.array([0.], dtype="float32")
        total_d_loss = np.array([0.], dtype="float32")
        total_acc = np.array([0.], dtype="float32")
        start = time.time()
        for data in dataset.get_data():
            data = toGPU(data, conf.gpu_num)

            if count % conf.n_dis  == 0:
                loss = updater.update_PG(data)
                total_g_loss += loss.data.cpu().numpy()
                
            loss, acc = updater.update_dis(data)
            total_d_loss += loss.data.cpu().numpy()
            total_acc += loss.data.cpu().numpy()

            count += 1
            if dataset.now_iter % conf.display_interval == 0:
                elapsed = time.time() - start
                pprog(elapsed, dataset.get_state)
                start = time.time()

        loss_hist[0,i] = total_d_loss / count
        loss_hist[1,i] = total_acc / count
        loss_hist[2,i] = total_g_loss / (count // 5)
        
        if i % conf.snapshot_interval == 0 and not conf.silent:
            sample_generate(netG, SAMPLE_SIZE, index2tok, preview_path + f"/sample_text_{i:04d}.txt")
            np.save(save_path + "/loss_hist", loss_hist)
            torch.save(netG.state_dict(), save_path + f"/gen_params_{i:04d}.pth")
            torch.save(netD.state_dict(), save_path + f"/dis_params_{i:04d}.pth")

    if not conf.silent:
        np.save(save_path + "/loss_hist", loss_hist)
        sample_generate(netG, SAMPLE_SIZE, index2tok, preview_path + "/sample_text.txt")
        torch.save(netG.state_dict(), save_path + "/gen_params.pth")
        torch.save(netD.state_dict(), save_path + "/dis_params.pth")
    print("\n\n\n\n==========================================")
    print("Info:finish train")

if __name__ == "__main__":
    main()