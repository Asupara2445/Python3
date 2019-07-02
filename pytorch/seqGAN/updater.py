import torch
import torch.nn as nn
from torch.autograd import Variable


def gen_data_prepare(text, start_letter=0, gpu=True):
    batch_size, seq_len = text.size()

    inp = torch.zeros(batch_size, seq_len)
    target = text
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len-1]

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target

def prepare_discriminator_data(pos_samples, neg_samples, gpu=True):
    inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0

    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    inp = Variable(inp)
    target = Variable(target).view(-1,1)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


class Updater():
    def __init__(self, netG, netD, optimizerG, optimizerD, conf):
        self.netG = netG
        self.netD = netD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.n_dis = conf.n_dis
        self.gpu_num = conf.gpu_num
        self.batch_size = conf.batch_size
        self.sample_size = conf.batch_size//conf.gpu_num
        self.gpu = False

        if self.gpu_num > 0:
            self.gpu = True
        
        if conf.batch_size % conf.gpu_num != 0:
            raise "batch size must be multiples of the number of GPU(s)."


    def update_pre_gen(self, data):
        self.netG.train()

        text = getattr(data, "text")
        inp, target = gen_data_prepare(text, gpu=self.gpu)
        self.netG.zero_grad()

        loss = self.netG(inp, target=target)
        if self.gpu_num > 1:
            loss = loss.sum()

        loss.backward()
        self.optimizerG.step()

        return loss / self.batch_size


    def update_dis(self, data):
        self.netD.train()

        text = getattr(data, "text")

        fake_text = self.netG(self.sample_size)
        inp, target = prepare_discriminator_data(text, fake_text, gpu=self.gpu)

        self.netD.zero_grad()

        loss, acc = self.netD(inp, target)
        if self.gpu_num > 1:
            loss = loss.sum()
            acc = acc.sum() / self.gpu_num

        loss.backward()
        self.optimizerD.step()

        return loss / self.batch_size, acc / self.batch_size
    

    def update_PG(self, data):
        g_loss, d_loss = 0, 0

        text = getattr(data, "text")

        fake_text = self.netG(self.sample_size)
        inp, target = gen_data_prepare(fake_text, gpu=self.gpu)
        rewards = self.netD(target)

        self.netG.zero_grad()
        loss = self.netG(inp, target, rewards)
        if self.gpu_num > 1:
            loss = loss.sum()

        loss.backward()
        self.optimizerG.step()

        return loss / self.batch_size

    def evaluate(self, data):
        g_loss, d_loss = 0, 0

        return g_loss, d_loss