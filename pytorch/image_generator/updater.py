import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

image_key = ["x_64", "x_128", "x_256"]


def to_img_dict_(inputs, img_num):
    
    res = {}
    count = 0
    for i in range(img_num):
        res[image_key[i]] = inputs[i]
        count += 1

    mean_var = (inputs[count], inputs[count+1])
    loss = mean_var

    return res, loss

def get_KL_Loss(mu, logvar):
    kld = mu.pow(2).add(logvar.mul(2).exp()).add(-1).mul(0.5).add(logvar.mul(-1))
    kl_loss = torch.mean(kld)
    return kl_loss

def compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels):

    criterion = nn.MSELoss()
    real_d_loss = criterion(real_logit, real_labels)
    wrong_d_loss = criterion(wrong_logit, fake_labels)
    fake_d_loss = criterion(fake_logit, fake_labels)

    discriminator_loss = real_d_loss + (wrong_d_loss+fake_d_loss) / 2.
    return discriminator_loss

def compute_d_img_loss(wrong_img_logit, real_img_logit, fake_img_logit, real_labels, fake_labels):

    criterion = nn.MSELoss()
    wrong_d_loss = criterion(wrong_img_logit, real_labels)
    real_d_loss  = criterion(real_img_logit, real_labels)
    fake_d_loss  = criterion(fake_img_logit, fake_labels)

    return fake_d_loss + (wrong_d_loss+real_d_loss) / 2

def compute_g_loss(fake_logit, real_labels):

    criterion = nn.MSELoss()
    generator_loss = criterion(fake_logit, real_labels)
    return generator_loss


class Updater():
    def __init__(self, netG, netD, optimizerG, optimizerD, conf):
        self.netG = netG
        self.netD = netD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.n_dis = conf.n_dis
        self.kl_coef = conf.kl_coef
        self.gpu_num = conf.gpu_num
        self.noise_dim = conf.noise_dim
        self.batch_size = conf.batch_size
        self.side_output_at = conf.side_output_at

        self.loss_hist = np.zeros((2,conf.max_epoch))

        self.noise = Variable(torch.FloatTensor(self.batch_size, self.noise_dim))

        self.REAL_GLOBAL_LABELS = Variable(torch.FloatTensor(self.batch_size, 1).fill_(1))
        self.FAKE_GLOBAL_LABELS = Variable(torch.FloatTensor(self.batch_size, 1).fill_(0))

        self.REAL_LOCAL_LABELS = Variable(torch.FloatTensor(self.batch_size, 1, 5, 5).fill_(1))
        self.FAKE_LOCAL_LABELS = Variable(torch.FloatTensor(self.batch_size, 1, 5, 5).fill_(0))

        if self.gpu_num > 0:
            self.noise = self.noise.cuda()
            self.REAL_GLOBAL_LABELS = self.REAL_GLOBAL_LABELS.cuda()
            self.FAKE_GLOBAL_LABELS = self.FAKE_GLOBAL_LABELS.cuda()
            self.REAL_LOCAL_LABELS = self.REAL_LOCAL_LABELS.cuda()
            self.FAKE_LOCAL_LABELS = self.FAKE_LOCAL_LABELS.cuda()

    def get_labels(self, logit):
        batch_size = logit.size()[0]

        if logit.size(-1) == 1: 
            real_labels = self.REAL_GLOBAL_LABELS
            fake_labels = self.FAKE_GLOBAL_LABELS

        else:
            real_labels = self.REAL_LOCAL_LABELS
            fake_labels = self.FAKE_LOCAL_LABELS

        return real_labels.view_as(logit), fake_labels.view_as(logit)

    def update(self, data, i):
        for p in self.netD.parameters(): p.requires_grad = True
        self.netD.zero_grad()

        images = getattr(data, "image")
        wrong_images = getattr(data, "wrong_image")
        embeddings = getattr(data, "en_text_feature")

        self.noise.data.normal_(0, 1)
        fake_images, mean_var = to_img_dict_(self.netG(embeddings, self.noise), len(self.side_output_at))
        d_loss = 0
        for key in fake_images:
            this_img = images[key].detach()
            this_wrong = wrong_images[key]
            this_fake = Variable(fake_images[key].data)

            real_logit,  real_img_logit_local = self.netD(this_img, embeddings)
            wrong_logit, wrong_img_logit_local = self.netD(this_wrong, embeddings)
            fake_logit,  fake_img_logit_local = self.netD(this_fake, embeddings)

            """ compute disc pair loss """
            real_labels, fake_labels = self.get_labels(real_logit)
            pair_loss =  compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)

            """ compute disc image loss """
            real_labels, fake_labels = self.get_labels(real_img_logit_local)
            img_loss = compute_d_img_loss(wrong_img_logit_local, real_img_logit_local, fake_img_logit_local, real_labels, fake_labels)

            d_loss += (pair_loss + img_loss)
        d_loss.backward()
        self.optimizerD.step()
        
        for p in self.netD.parameters(): p.requires_grad = False
        self.netG.zero_grad()

        g_loss = self.kl_coef * get_KL_Loss(mean_var[0], mean_var[1])
        """Compute gen loss"""
        for key in fake_images:
            this_fake = fake_images[key]
            fake_pair_logit, fake_img_logit_local = self.netD(this_fake, embeddings)

            # -- compute pair loss ---
            real_labels, _ = self.get_labels(fake_pair_logit)
            g_loss += compute_g_loss(fake_pair_logit, real_labels)

            # -- compute image loss ---
            real_labels, _ = self.get_labels(fake_img_logit_local)
            img_loss = compute_g_loss(fake_img_logit_local, real_labels)
            g_loss += img_loss

        g_loss.backward()
        self.optimizerG.step()

        self.loss_hist[0,i] = float(g_loss.data.cpu().numpy())
        self.loss_hist[1,i] = float(d_loss.data.cpu().numpy())
        #torch.cuda.empty_cache()
    
    def evaluate(self, data):
        self.netG.eval()
        g_loss, d_loss = 0, 0

        return g_loss, d_loss