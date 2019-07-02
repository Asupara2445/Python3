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

def get_noise(batch_size, noise_dim, gpu):
    noise = Variable(torch.FloatTensor(batch_size, noise_dim))
    noise.data.normal_(0, 1)

    if gpu:noise = noise.cuda()

    return noise

def get_rabels(batch_size, gpu):
    real_labels = Variable(torch.FloatTensor(batch_size, 1).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size, 1).fill_(0))

    if gpu:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
    
    return real_labels, fake_labels


class Updater():
    def __init__(self, netG, netD, optimizerG, optimizerD, conf):
        self.netG = netG
        self.netD = netD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.clip = conf.clip
        self.noise_dim = conf.noise_dim
        self.n_rollout = conf.n_rollout
        self.gpu_num = conf.gpu_num
        self.gpu = False

        if self.gpu_num > 0:
            self.gpu = True
            
        if conf.batch_size % conf.gpu_num != 0:
            raise "batch size must be multiples of the number of GPU(s)."


    def update_pre_gen(self, data):
        for p in self.netG.parameters(): p.requires_grad = True
        self.netG.zero_grad()

        text = getattr(data, "text")
        image = getattr(data, "image")
        inp, target = gen_data_prepare(text, gpu=self.gpu)
        batch_size = text.size()[0]
        noise = get_noise(batch_size, self.noise_dim, self.gpu)

        loss = self.netG(inp, [image, noise], target=target)
        if self.gpu_num > 1:
            loss = loss.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.clip)
        self.optimizerG.step()
        #torch.cuda.empty_cache()

        return loss / batch_size


    def update_dis(self, data):
        for p in self.netD.parameters(): p.requires_grad = True
        self.netD.zero_grad()

        text = getattr(data, "text")
        wrong_text = getattr(data, "wrong_text")
        image = getattr(data, "image")

        batch_size = text.size()[0]
        noise = get_noise(batch_size, self.noise_dim, self.gpu)
        real_labels , fake_labels = get_rabels(batch_size,  self.gpu)

        sample_size = batch_size if self.gpu_num < 2 else batch_size // self.gpu_num
        fake_text = self.netG(sample_size, [image, noise])
        fake_text = Variable(fake_text.data)

        real_loss, real_acc = self.netD([text, image], real_labels)
        fake_loss, fake_acc = self.netD([fake_text, image], fake_labels)
        wrong_loss, wrong_acc = self.netD([wrong_text, image], fake_labels)

        if self.gpu_num > 1:
            real_loss = real_loss.sum();real_acc = real_acc.sum() / self.gpu_num
            fake_loss = fake_loss.sum();fake_acc = fake_acc.sum() / self.gpu_num
            wrong_loss = wrong_loss.sum();wrong_acc = wrong_acc.sum() / self.gpu_num
        
        loss = real_loss + (fake_loss + wrong_loss) / 2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netD.parameters(), self.clip)
        self.optimizerD.step()
        #torch.cuda.empty_cache()

        return loss / batch_size, real_acc / batch_size,\
                    fake_acc / batch_size, wrong_acc / batch_size
    

    def update_PG(self, data):
        for p in self.netD.parameters(): p.requires_grad = False
        self.netG.zero_grad()

        text = getattr(data, "text")
        image = getattr(data, "image")
        batch_size = text.size()[0]
        noise = get_noise(batch_size, self.noise_dim, self.gpu)

        sample_size = batch_size if self.gpu_num < 2 else batch_size // self.gpu_num

        loss = 0
        for i in range(self.n_rollout):
            fake_text = self.netG(sample_size, [image, noise])
            inp, target = gen_data_prepare(fake_text, gpu=self.gpu)
            rewards = self.netD([target, image])
            loss += self.netG(inp, [image, noise], target, rewards)
        loss = loss / self.n_rollout

        if self.gpu_num > 1:
            loss = loss.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.clip)
        self.optimizerG.step()
        #torch.cuda.empty_cache()

        return loss / batch_size


    def evaluate(self, data):

        text = getattr(data, "text")
        wrong_text = getattr(data, "wrong_text")
        image = getattr(data, "image")

        batch_size = text.size()[0]
        noise = get_noise(batch_size, self.noise_dim, self.gpu)
        real_labels , fake_labels = get_rabels(batch_size,  self.gpu)

        sample_size = batch_size if self.gpu_num < 2 else batch_size // self.gpu_num
        fake_text = self.netG(sample_size, [image, noise])

        inp, target = gen_data_prepare(fake_text, gpu=self.gpu)
        rewards = self.netD([target, image])
        g_loss = self.netG(inp, [image, noise], target, rewards)

        fake_text = Variable(fake_text.data)
        real_loss, real_acc = self.netD([text, image], real_labels)
        fake_loss, fake_acc = self.netD([fake_text, image], fake_labels)
        wrong_loss, wrong_acc = self.netD([wrong_text, image], fake_labels)

        if self.gpu_num > 1:
            g_loss = g_loss.sum()
            real_loss = real_loss.sum();real_acc = real_acc.sum() / self.gpu_num
            fake_loss = fake_loss.sum();fake_acc = fake_acc.sum() / self.gpu_num
            wrong_loss = wrong_loss.sum();wrong_acc = wrong_acc.sum() / self.gpu_num
        
        d_loss = real_loss + (fake_loss + wrong_loss) / 2
        #torch.cuda.empty_cache()

        return g_loss, d_loss, real_acc, fake_acc, wrong_acc