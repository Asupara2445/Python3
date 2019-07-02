import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable, serializers
from chainer.backends import cuda
import cupy as xp

class Updater(chainer.Chain):
    def __init__(self, model, netG_opt, netD_opt, netRNN_opt, n_dis, batch_size, gpu_num, max_iter):
        self.model = model
        self.netG_opt = netG_opt
        self.netD_opt = netD_opt
        self.netRNN_opt = netRNN_opt
        self.n_dis = n_dis
        self.batch_size = batch_size
        self.gpu_num = gpu_num
        self.max_iter = max_iter
        self.counter = 0
        self.loss_hist = np.zeros((3, self.max_iter))

    def add_loss_to_hist(self, d_loss, g_loss, r_loss):
        if self.loss_hist[0,self.counter] == 0:
            self.loss_hist[0,self.counter] = float(cuda.to_cpu(d_loss.data))
            self.loss_hist[1,self.counter] = float(cuda.to_cpu(g_loss.data))
            self.loss_hist[2,self.counter] = float(cuda.to_cpu(r_loss.data))
        else:
            self.loss_hist[0,self.counter] = (float(cuda.to_cpu(d_loss.data)) + self.loss_hist[0,self.counter]) / 2
            self.loss_hist[1,self.counter] = (float(cuda.to_cpu(g_loss.data)) + self.loss_hist[1,self.counter]) / 2
            self.loss_hist[2,self.counter] = (float(cuda.to_cpu(r_loss.data)) + self.loss_hist[2,self.counter]) / 2
        
    def update(self, data, now_epoch):
        for i in range(self.gpu_num):
            netG = getattr(self.model, f"netG_{i}")
            netD = getattr(self.model, f"netD_{i}")
            netRNN = getattr(self.model, f"netRNN_{i}")

            real_img = getattr(data, f"image_{i}")
            text = getattr(data, f"text_{i}")
            wrong_img = getattr(data, f"wrong_image_{i}")
            wrong_text = getattr(data, f"wrong_text_{i}")

            embedding, w_embedding, r_loss = netRNN(real_img, text, wrong_img, wrong_text)
            setattr(self, f"r_loss_{i}", r_loss)
            setattr(self, f"embedding_{i}", embedding)
            setattr(self, f"w_embedding_{i}", w_embedding)

            fake_img = netG(embedding)
            real_logits = netD(real_img, embedding)
            fake_logits = netD(fake_img, embedding)
            wrong_logits = netD(real_img, w_embedding)

            setattr(self, f"fake_img_{i}", fake_img)

            setattr(self, f"real_logits_{i}", real_logits)
            setattr(self, f"fake_logits_{i}", fake_logits)
            setattr(self, f"wrong_logits_{i}", wrong_logits)

        for i in range(self.gpu_num):
            netG = getattr(self.model, f"netG_{i}")
            netD = getattr(self.model, f"netD_{i}")
            netRNN = getattr(self.model, f"netRNN_{i}")

            fake_img = getattr(self, f"fake_img_{i}")

            embedding = getattr(self, f"embedding_{i}")
            w_embedding = getattr(self, f"w_embedding_{i}")
            if now_epoch <= 50:
                r_loss = getattr(self, f"r_loss_{i}")
                netRNN.cleargrads()
                r_loss.backward()
            embedding.unchain_backward()
            w_embedding.unchain_backward()

            real_logits = getattr(self, f"real_logits_{i}")
            fake_logits = getattr(self, f"fake_logits_{i}")
            wrong_logits = getattr(self, f"wrong_logits_{i}")

            ones = cuda.to_gpu(xp.ones_like(real_logits.data, dtype="int32"), i)
            zeros = cuda.to_gpu(xp.zeros_like(real_logits.data, dtype="int32"), i)

            if self.counter % self.n_dis == 0:
                g_loss = F.sigmoid_cross_entropy(fake_logits, ones)
                netG.cleargrads()
                g_loss.backward()
            fake_img.unchain_backward()
            
            d_loss1 = F.sigmoid_cross_entropy(real_logits, ones)
            d_loss2 = F.sigmoid_cross_entropy(fake_logits, zeros)
            d_loss3 = F.sigmoid_cross_entropy(wrong_logits, zeros)

            d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5
            netD.cleargrads()
            d_loss.backward()

            self.add_loss_to_hist(d_loss, g_loss, r_loss)

        #add calc grad 
        netG_0 = getattr(self.model, "netG_0")
        netD_0 = getattr(self.model, "netD_0")
        for i in range(1,self.gpu_num-1):
            netG = getattr(self.model, f"netG_{i}")
            netD = getattr(self.model, f"netD_{i}")
            netG_0.addgrads(netG)
            netD_0.addgrads(netD)
        
        if now_epoch <= 50:
            netRNN_0 = getattr(self.model, f"netRNN_0")
            for i in range(1,self.gpu_num-1):
                netRNN = getattr(self.model, f"netRNN_{i}")
                netRNN_0.addgrads(netRNN)
            self.netRNN_opt.update()
            
        self.netG_opt.update()
        self.netD_opt.update()
        cuda.memory_pool.free_all_blocks()
        self.counter += 1