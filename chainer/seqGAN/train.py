import os

import chainer
import cupy as xp
import numpy as np
from chainer import iterators, training
from chainer.training import extensions
from chainer.backends import cuda

from dataset.dataset import get_Dataset
from models.generator import Generator
from models.discriminator import Discriminator
from updater import Updater
from utils.configure import get_conf


def build_models(conf, n_voc, seq_len):
    model = dict()

    netG = Generator(n_voc, conf.emb_dim, conf.hid_dim, seq_len, conf.gpu_num)
    netD = Discriminator(n_voc, conf.emb_dim, conf.hid_dim, seq_len, conf.gpu_num)

    if conf.gpu_num == 0:
        model["netG_0"] = netG
        model["netD_0"] = netD

    else:
        for i in range(conf.gpu_num):
            copy_netG = netG.copy()
            copy_netD = netD.copy()

            copy_netG.gpu_id = i
            copy_netD.gpu_id = i

            model[f"netG_{i}"] = copy_netG.to_gpu(i)
            model[f"netD_{i}"] = copy_netD.to_gpu(i)
    
    return model
        
def build_opts(model, alpha, beta1, beta2):
    optimizer = dict()

    netG = model["netG_0"]
    netD =  model["netD_0"]

    netG_opt = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    netG_opt.setup(netG)

    netD_opt = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    netD_opt.setup(netD)

    optimizer["netG_opt"] = netG_opt
    optimizer["netD_opt"] = netD_opt

    return optimizer

def main():
    conf = get_conf()
    train, val, index2tok, SEQ_LEN = get_Dataset(conf)
    train_iter = iterators.SerialIterator(train, conf.batch_size)
    val_iter = iterators.SerialIterator(val, conf.batch_size)

    N_VOC =  len(list(index2tok.keys()))

    if conf.gpu_num > 0:
        for i in range(conf.gpu_num):
            cuda.get_device_from_id(i).use()

    models = build_models(conf, N_VOC, SEQ_LEN)
    opts = build_opts(models, conf.adam_alpha, conf.adam_beta1, conf.adam_beta2)

    updater = Updater(train_iter, models, opts, conf.gpu_num)
    trainer = training.Trainer(updater, (conf.max_epoch, "epoch"), out=conf.save_path)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == "__main__":
    main()