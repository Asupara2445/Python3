import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from PIL import Image

from dataset.mscoco import MSCOCO
from models.generator import GENERATOR
from models.discriminator import DISCRIMINATOR
from utils.configure import get_args
from utils.data_downloader import download_file_from_google_drive
from utils.progress import print_progress, progress
from updater import Updater

script_path = os.path.dirname(__file__)
image_keys = ["x_64", "x_128", "x_256"]

def toGPU(batch, gpu_num):
    class Data():
        def __init__(self):pass
    
    data = Data

    if gpu_num < 0:
        image = {key:torch.tensor([data[key] for data in batch[0]]) for key in image_keys}
        en_text_feature = torch.tensor(batch[1])
        
        wrong_image = {key:torch.tensor([data[key] for data in batch[2]]) for key in image_keys}
        wrong_en_text_feature = torch.tensor(batch[3])
    else:
        image = {key:torch.tensor([data[key] for data in batch[0]]).cuda() for key in image_keys}
        en_text_feature = torch.tensor(batch[1]).cuda()

        wrong_image = {key:torch.tensor([data[key] for data in batch[2]]).cuda() for key in image_keys}
        wrong_en_text_feature = torch.tensor(batch[3]).cuda()
    
    setattr(data, "image", image)
    setattr(data, "en_text_feature", en_text_feature)
    setattr(data, "wrong_image", wrong_image)
    setattr(data, "wrong_en_text_feature", wrong_en_text_feature)
    
    return data

def build_models(conf):
    side_output_at = conf.side_output_at
    netG = GENERATOR(conf.sent_dim, conf.noise_dim, conf.emb_dim, conf.hid_dim, conf.n_resblock, side_output_at=side_output_at)
    netD = DISCRIMINATOR(3, conf.hid_dim, conf.sent_dim, conf.emb_dim, side_output_at=side_output_at)

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

def sample_generate(netG, data, conf):
    netG.eval()
    real_img = np.asarray([img["x_256"] for img in data[0]])
    noise = Variable(torch.FloatTensor(conf.sample_size, conf.noise_dim))
    sent_emb = torch.tensor(data[1])

    if conf.gpu_num > 0:
        noise = noise.cuda()
        sent_emb = sent_emb.cuda()
    noise.data.normal_(0,1)

    _, _, fake_imgs, _, _ = netG(sent_emb, noise)

    if conf.gpu_num > 0:
        fake_img = fake_imgs.data.cpu().numpy()
    else:
        fake_img = fake_imgs.data.numpy()

    fake_img = np.asanyarray(np.clip(fake_img * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    real_img = np.asanyarray(np.clip(real_img * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    x = np.concatenate([fake_img, real_img], axis=0)
    _, _, h, w = x.shape
    x = x.reshape((2, conf.sample_size, 3, h, w))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((2 * h, conf.sample_size * w, 3))

    return x

def main():
    conf = get_args()

    if not conf.silent:
        save_path = os.path.abspath(script_path + conf.save_path)
        if not os.path.exists(save_path):os.mkdir(save_path)
        preview_path = os.path.abspath(save_path + "/preview")
        if not os.path.exists(preview_path):os.mkdir(preview_path)

    dataset = MSCOCO(conf)
    netG, netD = build_models(conf)
    optimizerG, optimizerD = build_optimizer(netG, netD, conf.adam_lr, conf.adam_beta1, conf.adam_beta2)
    pprog = print_progress(conf.max_epoch, conf.batch_size, dataset.train_data_len, use_epoch=True)

    updater = Updater(netG, netD, optimizerG, optimizerD, conf)
    
    print("==========================================")
    print("Info:start train")
    
    val_times = dataset.val_data_len // dataset.batch_size
    if dataset.val_data_len % dataset.batch_size != 0:val_times += 1
    for i in range(conf.max_epoch):
        train_loss = np.array([0.,0.], dtype="float32")
        start = time.time()
        for data in dataset.get_data():
            data = toGPU(data, conf.gpu_num)
            updater.update(data, i)

            if dataset.now_iter % conf.display_interval == 0:
                elapsed = time.time() - start
                pprog(elapsed, dataset.get_state)
                start = time.time()
        
        if i % conf.snapshot_interval == 0 and not conf.silent:
            data = dataset.sampling(conf.sample_size)
            sample = sample_generate(netG, data, conf)
            Image.fromarray(sample).save(preview_path + f"/image_{i:04d}.png")
            
    print("\n\n\n\n==========================================")
    print("Info:finish train")

if __name__ == "__main__":
    main()