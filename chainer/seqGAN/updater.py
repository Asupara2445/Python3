import chainer
import chainer.functions as F
from chainer import training, reporter
from chainer.backends import cuda
from chainer.dataset import iterator as iterator_module

import cupy as xp
import numpy as np


def converter(batch, device):
    if device == 0:
        batch = np.array(batch)

    elif device == 1:
        batch = cuda.to_gpu(xp.array(batch))

    elif device >= 2:
        batch = cuda.to_cpu(batch)
        batch = xp.split(xp.array(batch).astype(xp.int64), device)
        batch = [cuda.to_gpu(batch[i], i) for i in range(device)]
    
    return batch

def prepare_generator_data(data, device):
    data = data.T
    if device >= 1:
        inp = xp.zeros_like(data)
    else:
        data = np.asanyarray(data).astype(np.int64)
        inp = np.zeros_like(data)

    seq_len, _ = data.shape
    for i in range(seq_len-1):
        inp[i+1] = data[i]

    data = data.T
    inp = inp.T

    return converter(inp, device), converter(data, device)

def prepare_discriminator_data(pos_samples, neg_samples, device):
    if device >= 1:
        pos_samples = xp.array(pos_samples)

    inp = xp.concatenate((pos_samples, neg_samples), 0)
    target = xp.ones(pos_samples.shape[0] + neg_samples.shape[0]).astype(np.int64)
    target[pos_samples.shape[0]:] = 0

    perm = xp.random.permutation(target.shape[0])
    target = target[perm]
    inp = inp[perm]

    target = target.reshape((-1,1))

    return converter(inp, device), converter(target, device)


class Updater(training.StandardUpdater):
    def __init__(self, iterator, models, optimizer, device):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self.models = models
        self._optimizers = optimizer
        self.device = device
        self.iteration = 0

        if self.device < 0:xp = np

    def update_core(self):
        netG_opt = self._optimizers["netG_opt"]
        netD_opt = self._optimizers["netD_opt"]

        iterator = np.array(self._iterators['main'].next())
        sample_size = len(iterator)

        netG = self.models["netG_0"]
        fake_text = netG(sample_size)

        self.g_loss = 0
        self.d_loss = 0
        self.accuracy = 0
        g_inp, g_target = prepare_generator_data(fake_text, self.device)
        d_inp, d_target = prepare_discriminator_data(iterator, fake_text, self.device)
        for i in range(self.device):
            netG = self.models[f"netG_{i}"]
            netD = self.models[f"netD_{i}"]
            
            rewards = netD(g_target[i])
            rewards.unchain_backward()

            g_loss = netG(g_inp[i], g_target[i], rewards)
            d_loss, acc = netD(d_inp[i], d_target[i])
            self.g_loss += cuda.to_cpu(g_loss.data)
            self.d_loss += cuda.to_cpu(d_loss.data)
            self.accuracy += cuda.to_cpu(acc.data)

            netG.cleargrads()
            netD.cleargrads()

            g_loss.backward()
            d_loss.backward()
        self.accuracy /= 2
        
        netG_0 = self.models["netG_0"]
        netD_0 = self.models["netD_0"]
        for i in range(1,self.device-1):
            netG = self.models[f"netG_{i}"]
            netD = self.models[f"netD_{i}"]
            netG_0.addgrads(netG)
            netD_0.addgrads(netD)

        netG_opt.update()
        netD_opt.update()
        cuda.memory_pool.free_all_blocks()

        return self.g_loss