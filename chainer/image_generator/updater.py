import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable, serializers
from chainer.backends import cuda
import cupy as xp

def compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels):

    real_d_loss = F.mean_squared_error(real_logit, real_labels)
    wrong_d_loss = F.mean_squared_error(wrong_logit, fake_labels)
    fake_d_loss = F.mean_squared_error(fake_logit, fake_labels)

    return real_d_loss + (wrong_d_loss+fake_d_loss) / 2.

def compute_d_img_loss(wrong_img_logit, real_img_logit, fake_img_logit, real_labels, fake_labels):

    wrong_d_loss = F.mean_squared_error(wrong_img_logit, real_labels)
    real_d_loss  = F.mean_squared_error(real_img_logit, real_labels)
    fake_d_loss  = F.mean_squared_error(fake_img_logit, fake_labels)

    return fake_d_loss + (wrong_d_loss+real_d_loss) / 2

def compute_g_loss(fake_logit, real_labels):
    return F.mean_squared_error(fake_logit, real_labels)

def unchain_backward(image_dict):
    for key in image_dict.keys():
        image_dict[key].unchain_backward()

class Updater(chainer.Chain):
    def __init__(self, model, netG_opt, netD_opt, n_dis, batch_size, gpu_num, KL_loss_iter, KL_loss_conf, epoch_decay, max_iter):
        self.model = model
        self.netG_opt = netG_opt
        self.netD_opt = netD_opt
        self.n_dis = n_dis
        self.batch_size = batch_size
        self.gpu_num = gpu_num
        self.KL_loss_iter = KL_loss_iter
        self.KL_loss_conf = KL_loss_conf
        self.epoch_decay = epoch_decay
        self.max_iter = max_iter
        self.KL_loss_ratio = 0
        self.KL_counter = 0
        self.counter = 0
        self.now_epoch = 0
        self.loss_hist = np.zeros((8, self.max_iter))
        

    def add_loss_to_hist(self, real_pair_loss, real_local_loss, fake_pair_loss, fake_local_loss, wrong_pair_loss, wrong_local_loss, gen_KL_loss):
        if self.loss_hist[0,self.counter] == 0:
            self.loss_hist[0,self.counter] = float(cuda.to_cpu(real_pair_loss.data))
            self.loss_hist[1,self.counter] = float(cuda.to_cpu(real_local_loss.data))
            self.loss_hist[2,self.counter] = float(cuda.to_cpu(fake_pair_loss.data))
            self.loss_hist[3,self.counter] = float(cuda.to_cpu(fake_local_loss.data))
            self.loss_hist[4,self.counter] = float(cuda.to_cpu(wrong_pair_loss.data))
            self.loss_hist[5,self.counter] = float(cuda.to_cpu(wrong_local_loss.data))
            self.loss_hist[6,self.counter] = float(cuda.to_cpu(gen_KL_loss.data))
        else:
            self.loss_hist[0,self.counter] = (float(cuda.to_cpu(real_pair_loss.data)) + self.loss_hist[0,self.counter]) / 2
            self.loss_hist[1,self.counter] = (float(cuda.to_cpu(real_local_loss.data)) + self.loss_hist[1,self.counter]) / 2
            self.loss_hist[2,self.counter] = (float(cuda.to_cpu(fake_pair_loss.data)) + self.loss_hist[2,self.counter]) / 2
            self.loss_hist[3,self.counter] = (float(cuda.to_cpu(fake_local_loss.data)) + self.loss_hist[3,self.counter]) / 2
            self.loss_hist[4,self.counter] = (float(cuda.to_cpu(wrong_pair_loss.data)) + self.loss_hist[4,self.counter]) / 2
            self.loss_hist[5,self.counter] = (float(cuda.to_cpu(wrong_local_loss.data)) + self.loss_hist[5,self.counter]) / 2
            self.loss_hist[6,self.counter] = (float(cuda.to_cpu(gen_KL_loss.data)) + self.loss_hist[6,self.counter]) / 2
        
    def update(self, data, now_epoch):
        if self.KL_counter < self.KL_loss_iter:self.KL_loss_ratio = self.KL_counter * (1 / self.KL_loss_iter);self.KL_counter += 1
        else:self.KL_loss_ratio = 1

        for i in range(self.gpu_num):
            netG = getattr(self.model, f"netG_{i}")
            netD = getattr(self.model, f"netD_{i}")

            depth = getattr(data, f"depth_{i}")
            real_img = getattr(data, f"image_{i}")
            embeddings = getattr(data, f"text_{i}")

            wrong_img = getattr(data, f"wrong_image_{i}")
            wrong_depth = getattr(data, f"wrong_depth_{i}")
            #wrong_text = getattr(data, f"wrong_text_{i}")

            fake_img, KL_loss = netG(embeddings)
            g_loss = self.KL_loss_conf * KL_loss
            #g_loss = self.KL_loss_conf * self.KL_loss_ratio * KL_loss

            d_loss = 0
            for key in real_img.keys():
                real_logit,  real_img_logit_local = netD(real_img[key], embeddings, fg=fg, bg=bg)
                fake_logit,  fake_img_logit_local = netD(fake_img[key], embeddings, fg=fg, bg=bg)
                wrong_logit, wrong_img_logit_local = netD(wrong_img[key], embeddings, fg=fg, bg=bg)

                ''' compute disc pair loss '''
                real_labels = cuda.to_gpu(xp.ones_like(real_logit.data, dtype="float32"), i)
                fake_labels = cuda.to_gpu(xp.zeros_like(real_logit.data, dtype="float32"), i)
                pair_loss =  compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)

                ''' compute disc image loss '''
                real_labels = cuda.to_gpu(xp.ones_like(real_img_logit_local.data, dtype="float32"), i)
                fake_labels = cuda.to_gpu(xp.zeros_like(real_img_logit_local.data, dtype="float32"), i)
                img_loss = compute_d_img_loss(wrong_img_logit_local, real_img_logit_local, fake_img_logit_local, real_labels, fake_labels)

                d_loss += (pair_loss + img_loss)

                ''' compute gen loss '''
                real_labels = cuda.to_gpu(xp.ones_like(fake_logit.data, dtype="float32"), i)
                g_loss += compute_g_loss(fake_logit, real_labels)
                real_labels = cuda.to_gpu(xp.ones_like(fake_img_logit_local.data, dtype="float32"), i)
                g_loss += compute_g_loss(fake_img_logit_local, real_labels)

            if self.counter % self.n_dis == 0:
                netG.cleargrads()
                g_loss.backward()
            unchain_backward(fake_img)

            netD.cleargrads()
            d_loss.backward()

        #add calc grad 
        netG_0 = getattr(self.model, "netG_0")
        netD_0 = getattr(self.model, "netD_0")
        for i in range(1,self.gpu_num-1):
            netG = getattr(self.model, f"netG_{i}")
            netD = getattr(self.model, f"netD_{i}")
            netG_0.addgrads(netG)
            netD_0.addgrads(netD)
        
        if self.now_epoch != now_epoch:
            self.now_epoch = now_epoch
            if self.now_epoch % self.epoch_decay == 0:
                self.netG_opt.hyperparam.alpha /= 2
                self.netD_opt.hyperparam.alpha /= 2

        self.netG_opt.update()
        self.netD_opt.update()
        cuda.memory_pool.free_all_blocks()
        self.counter += 1