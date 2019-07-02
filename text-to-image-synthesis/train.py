import os
import json
import time
import numpy as np
from PIL import Image

import chainer
from chainer import serializers
from chainer.backends import cuda
import cupy as xp

from dataset.make_dataset import Dataset
from utils.progress import print_progress
from utils.get_args import get_args
from models.generator import Generator
from models.discriminator import Discriminator
from models.encoder import DS_SJE
from updater import Updater

script_path = os.path.dirname(__file__)

cuda.get_device_from_array(0)
cuda.get_device_from_array(1)

def toGPU(batch, gpu_num):
    class Data(): pass
    data = Data()

    if gpu_num == 0:
        setattr(data, "image_0", batch[0])
        setattr(data, "text_0", batch[1])
        setattr(data, "wrong_image_0", batch[2])
        setattr(data, "wrong_text_0", batch[3])

    else:
        split_size = len(batch[0]) // 2

        image = xp.array_split(xp.array(batch[0], dtype="float32"), gpu_num)
        text = [batch[1][i*split_size:(i+1)*split_size] for i in range(gpu_num)]

        wrong_image = xp.array_split(xp.array(batch[2], dtype="float32"), gpu_num)
        wrong_text = [batch[3][i*split_size:(i+1)*split_size] for i in range(gpu_num)]

        for i in range(gpu_num):
            setattr(data, f"image_{i}", cuda.to_gpu(image[i], i))
            setattr(data, f"text_{i}", cuda.to_gpu(text[i], i))
            setattr(data, f"wrong_image_{i}", cuda.to_gpu(wrong_image[i], i))
            setattr(data, f"wrong_text_{i}", cuda.to_gpu(wrong_text[i], i))

    return data

def make_model(args, dataset):
    N = args.gpu_num
    class Model(): pass
    model = Model()

    netG = Generator(args)
    netD = Discriminator(args)
    netRNN = DS_SJE(args, dataset.embed_mat)
    if N == 0:
        setattr(model, "netG_0", netG)
        setattr(model, "netD_0", netD)
        setattr(model, "netRNN_0", netRNN)
        
    else:
        for i in range(N):
            temp_netG = netG.copy()
            temp_netD = netD.copy()
            temp_netRNN = netRNN.copy()
            temp_netG.gpu_id = i
            temp_netD.gpu_id = i
            temp_netRNN.gpu_id = i
            setattr(model, f"netG_{i}", temp_netG.to_gpu(i))
            setattr(model, f"netD_{i}", temp_netD.to_gpu(i))
            setattr(model, f"netRNN_{i}", temp_netRNN.to_gpu(i))
        
    return model

def make_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer

def sample_generate(model, data, noise_dim, noise_dist, seed=0):
    real_image, text = data
    data_size = len(text)
    text = cuda.to_gpu(text, 0)

    netG = getattr(model, f"netG_0")
    netRNN = getattr(model, f"netRNN_0")

    embedding = netRNN.rnn_encoder(text)
    img = netG.generate(embedding)
    img = np.asanyarray(np.clip(img * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    real_image = np.asanyarray(np.clip(real_image * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    x = np.concatenate([img, real_image], axis=0)
    _, _, h, w = x.shape
    x = x.reshape((2, data_size, 3, h, w))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((2 * h, data_size * w, 3))

    return x

def main():
    args = get_args()
    dataset = Dataset(args.name, args.image_size, args.gpu_num, args.batch_size)

    if not args.silent:
        save_path = os.path.abspath(script_path + args.save_path)
        if not os.path.exists(save_path):os.mkdir(save_path)
        save_path = os.path.abspath(save_path + f"/{args.name}")
        if not os.path.exists(save_path):os.mkdir(save_path)
        preview_path = os.path.abspath(save_path + "/preview")
        if not os.path.exists(preview_path):os.mkdir(preview_path)
    
    if args.max_epoch is not None:
        epoch_iter = dataset.train_data_len // args.batch_size
        if dataset.train_data_len % args.batch_size != 0:epoch_iter += 1
        args.max_iter = args.max_epoch * epoch_iter

    progress = print_progress(args.max_iter, args.batch_size, dataset.train_data_len)

    if args.gpu_num != 0:
        cuda.get_device_from_array(xp.array([i for i in range(args.gpu_num)])).use()

    model = make_model(args, dataset)
    netG_opt = make_optimizer(model.netG_0, args.adam_alpha, args.adam_beta1, args.adam_beta2)
    netD_opt = make_optimizer(model.netD_0, args.adam_alpha, args.adam_beta1, args.adam_beta2)
    netRNN_opt = make_optimizer(model.netRNN_0, args.adam_alpha, args.adam_beta1, args.adam_beta2)

    updater = Updater(model, netG_opt, netD_opt, netRNN_opt, args.n_dis, args.batch_size, args.gpu_num, args.max_iter)

    print("==========================================")
    print("Info:start train")
    start = time.time()
    progress.max_iter = args.max_iter
    for i in range(args.max_iter):

        data = toGPU(dataset.next(), args.gpu_num)
        updater.update(data, dataset.now_epoch)

        if dataset.now_iter % args.display_interval == 0:
            elapsed = time.time() - start
            progress(elapsed, dataset.get_state)
            np.save(save_path + "/loss_hist.npy", updater.loss_hist)
            start = time.time()
        
        if dataset.now_iter % args.snapshot_interval == 0 and not args.silent:
            data = dataset.sampling(args.sample_size)
            sample = sample_generate(model, data, args.noise_dim, args.noise_dist)
            Image.fromarray(sample).save(preview_path + f"/image_{dataset.now_iter:08d}.png")
            serializers.save_npz(save_path + f"/Generator_{dataset.now_iter:08d}.npz",model.netG_0)
            serializers.save_npz(save_path + f"/Discriminator_{dataset.now_iter:08d}.npz",model.netD_0)
                    
    if not args.silent:
        data = dataset.sampling(args.sample_size)
        sample = sample_generate(model, data, args.noise_dim, args.noise_dist)
        Image.fromarray(sample).save(preview_path + f"/image_{dataset.now_iter:08d}.png")
        serializers.save_npz(save_path + f"/DS_SJE_{dataset.now_iter:08d}.npz",model.netRNN_0)
        serializers.save_npz(save_path + f"/Generator_{dataset.now_iter:08d}.npz",model.netG_0)
        serializers.save_npz(save_path + f"/Discriminator_{dataset.now_iter:08d}.npz",model.netD_0)

    print("\n\n\n==========================================")
    print("Info:finish train")


if __name__ == "__main__":
    try:main()
    except:
        import traceback
        traceback.print_exc()