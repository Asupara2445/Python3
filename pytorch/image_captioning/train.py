import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

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
        image = torch.tensor(batch[0])
        text = torch.tensor(batch[1])
        wrong_text = torch.tensor(batch[2])

    else:
        image = torch.tensor(batch[0]).cuda()
        text = torch.tensor(batch[1]).cuda()
        wrong_text = torch.tensor(batch[2]).cuda()

    setattr(data, "image", image)
    setattr(data, "text", text)
    setattr(data, "wrong_text", wrong_text)
    
    return data

def build_models(conf, n_voc, seq_len):
    netG = GENERATOR(n_voc, conf.emb_dim, conf.hid_dim, conf.noise_dim, conf.img_dim, seq_len, conf.gpu_num)
    netD = DISCRIMINATOR(n_voc, conf.emb_dim, conf.hid_dim, conf.img_dim, seq_len, conf.gpu_num)

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

def sample_generate(netG, data, sample_size, index2tok, gpu_num, noise_dim, save_path):
    ids, image, text = data
    image = torch.Tensor(image)
    
    if gpu_num > 1:
        noise = Variable(torch.FloatTensor(sample_size * gpu_num, noise_dim))
    else:
        noise = Variable(torch.FloatTensor(sample_size, noise_dim))

    if gpu_num > 0:
        image = image.cuda()
        noise = noise.cuda()

    samples = []
    for i in range(3):
        noise.data.normal_(0, 1)
        sample = netG(sample_size, [image, noise])

        if gpu_num > 0:
            sample = sample.data.cpu().numpy()
        else:
            sample = sample.data.numpy()

        samples.append(sample)

    samples = np.asarray(samples).transpose((1,0,2)).tolist()
    samples = [[" ".join([index2tok[idx]for idx in text]) for text in texts] for texts in samples]

    with open(save_path, "w", encoding="utf_8") as fp:
        samples = "\n\n".join([f"{id}\n" + "\n".join(sample) for sample, id in zip(samples, ids)])
        fp.write(samples)


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


    def pretrain_generatr():
        print("==========================================")
        print("Info:start genarator pre train")
        pre_gen_loss_hist = np.zeros((1, conf.pre_gen_max_epoch), dtype="float32")
        for i in range(conf.pre_gen_max_epoch):
            count = 0
            total_loss = 0
            start = time.time()
            for data in dataset.get_data():
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
            data = dataset.sample(conf.sample_size)
            sample_generate(netG, data, SAMPLE_SIZE, index2tok, conf.gpu_num,\
                        conf.noise_dim, preview_path + f"/sample_text_pretrain.txt")
            np.save(save_path + "/pre_gen_loss_hist", pre_gen_loss_hist)
            torch.save(netG.state_dict(), save_path + "/pretrain_gen_params")
        print("\n\n\n\n==========================================")
    
    def pretrain_discriminator():
        print("==========================================")
        print("Info:start discriminator pre train")
        dataset.clear_state()
        pprog.max_iter = conf.pre_dis_max_epoch
        pre_dis_hist = np.zeros((4, conf.pre_dis_max_epoch), dtype="float32")
        for i in range(conf.pre_dis_max_epoch):
            count = 0
            total_loss = 0
            total_real_acc = 0
            total_fake_acc = 0
            total_wrong_acc =0
            start = time.time()
            for data in dataset.get_data():
                data = toGPU(data, conf.gpu_num)
                loss, real_acc, fake_acc, wrong_acc = updater.update_dis(data)

                total_loss += loss.data.cpu().numpy()
                total_real_acc += real_acc.data.cpu().numpy()
                total_fake_acc += fake_acc.data.cpu().numpy()
                total_wrong_acc += wrong_acc.data.cpu().numpy()

                count += 1
                if dataset.now_iter % conf.display_interval == 0:
                    elapsed = time.time() - start
                    pprog(elapsed, dataset.get_state)
                    start = time.time()

            pre_dis_hist[0,i] = total_loss / count
            pre_dis_hist[1,i] = total_real_acc / count
            pre_dis_hist[2,i] = total_fake_acc / count
            pre_dis_hist[3,i] = total_wrong_acc / count

        if not conf.silent:
            np.save(save_path + "/pre_dis_hist", pre_dis_hist)
            torch.save(netD.state_dict(), save_path + "/pretrain_dis_params")
        print("\n\n\n\n==========================================")


    if os.path.exists(save_path + "/pretrain_gen_params"):
        netG.load_state_dict(torch.load(save_path + "/pretrain_gen_params"))
    else:pretrain_generatr()

    if os.path.exists(save_path + "/pretrain_dis_params"):
        netD.load_state_dict(torch.load(save_path + "/pretrain_dis_params"))
    else:pretrain_discriminator()


    print("==========================================")
    print("Info:start main train")
    dataset.clear_state()
    pprog.max_iter = conf.max_epoch
    train_loss_hist = np.zeros((5, conf.max_epoch), dtype="float32")
    val_loss_hist = np.zeros((5, conf.max_epoch), dtype="float32")
    val_count = dataset.val_data_len // conf.batch_size
    if dataset.val_data_len % conf.batch_size != 1:val_count += 1
    for i in range(conf.max_epoch):
        #train loop
        count = 1
        total_g_loss = 0
        total_d_loss = 0
        total_real_acc = 0
        total_fake_acc = 0
        total_wrong_acc =0
        start = time.time()

        for p in netG.parameters(): p.requires_grad = True
        for p in netD.parameters(): p.requires_grad = True
        for data in dataset.get_data():
            data = toGPU(data, conf.gpu_num)

            if count % conf.n_dis  == 0:
                loss = updater.update_PG(data)
                total_g_loss += loss.data.cpu().numpy()
                
            loss, real_acc, fake_acc, wrong_acc = updater.update_dis(data)

            total_d_loss += loss.data.cpu().numpy()
            total_real_acc += real_acc.data.cpu().numpy()
            total_fake_acc += fake_acc.data.cpu().numpy()
            total_wrong_acc += wrong_acc.data.cpu().numpy()

            count += 1
            if dataset.now_iter % conf.display_interval == 0:
                elapsed = time.time() - start
                pprog(elapsed, dataset.get_state)
                start = time.time()

        train_loss_hist[0,i] = total_d_loss / count
        train_loss_hist[1,i] = total_real_acc / count
        train_loss_hist[2,i] = total_fake_acc / count
        train_loss_hist[3,i] = total_wrong_acc / count
        train_loss_hist[4,i] = total_g_loss / (count // 5)
        print("\n\n\n")

        #val loop
        print(f"Validation {i+1} / {conf.max_epoch}")
        count = 0
        total_g_loss = 0
        total_d_loss = 0
        total_real_acc = 0
        total_fake_acc = 0
        total_wrong_acc =0
        start = time.time()
        for p in netG.parameters(): p.requires_grad = False
        for p in netD.parameters(): p.requires_grad = False
        for data in dataset.get_data(is_val=True):
            data = toGPU(data, conf.gpu_num)
            
            g_loss, d_loss, real_acc, fake_acc, wrong_acc = updater.evaluate(data)

            count += 1
            if dataset.now_iter % conf.display_interval == 0:
                elapsed = time.time() - start
                progress(count+1, val_count, elapsed)

        progress(count, val_count, elapsed)
        val_loss_hist[0,i] = total_d_loss / count
        val_loss_hist[1,i] = total_real_acc / count
        val_loss_hist[2,i] = total_fake_acc / count
        val_loss_hist[3,i] = total_wrong_acc / count
        val_loss_hist[4,i] = total_g_loss / (count // 5)
        print("\u001B[5A", end="")
        
        if (i+1) % conf.snapshot_interval == 0 and not conf.silent:
            data = dataset.sample(conf.sample_size)
            sample_generate(netG, data, SAMPLE_SIZE, index2tok, conf.gpu_num,\
                        conf.noise_dim, preview_path + f"/sample_text_{i+1:04d}.txt")
            np.save(save_path + "/train_loss_hist", train_loss_hist)
            np.save(save_path + "/val_loss_hist", val_loss_hist)
            torch.save(netG.state_dict(), save_path + f"/gen_params_{i+1:04d}.pth")
            torch.save(netD.state_dict(), save_path + f"/dis_params_{i+1:04d}.pth")

    if not conf.silent:
        np.save(save_path + "/train_loss_hist", train_loss_hist)
        np.save(save_path + "/val_loss_hist", val_loss_hist)
        data = dataset.sample(conf.sample_size)
        sample_generate(netG, data, SAMPLE_SIZE, index2tok, conf.gpu_num,\
                    conf.noise_dim, preview_path + "/sample_text.txt")
        torch.save(netG.state_dict(), save_path + "/gen_params.pth")
        torch.save(netD.state_dict(), save_path + "/dis_params.pth")
    print("\n\n\n\n==========================================")
    print("Info:finish train")

if __name__ == "__main__":
    main()