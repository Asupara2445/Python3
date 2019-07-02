import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd

import torchvision.models as models


class GENERATOR(nn.Module):
    def __init__(self, n_voc, emb_dim, hid_dim, noise_dim, img_dim, seq_len, gpu_num, oracle_init=False):
        super(GENERATOR, self).__init__()
        self.n_voc = n_voc
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.noise_dim = noise_dim
        self.img_dim = img_dim
        self.seq_len = seq_len
        self.gpu_num = gpu_num

        self.encoder = nn.Linear(4096, self.img_dim)
        self.embed2input = nn.Linear(self.noise_dim+self.img_dim, self.emb_dim)
        self.embeddings = nn.Embedding(n_voc, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)
        self.gru2out = nn.Linear(hid_dim, n_voc)

        if oracle_init:
            for p in self.parameters():
                init.normal(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hid_dim))

        if self.gpu_num > 0:
            return h.cuda()
        else:
            return h

    def forward(self, inp, cond, target=None, reward=None):
        img, noise = cond
        embed_img = self.encoder(img)
        x0 = torch.cat([embed_img, noise], dim=1)
        x0 = F.relu(self.embed2input(x0))

        if target is None and reward is None:
            sample_size = inp
            return self.sample(sample_size, x0)

        else:
            batch_size, seq_len = inp.size()
            inp = inp.permute(1, 0)
            target = target.permute(1, 0)
            h = self.init_hidden(batch_size)

            loss = 0
            if reward is None:
                loss_fn = nn.NLLLoss()
                _, h = self.calc_cond(x0, h)
                for i in range(seq_len):
                    out, h = self.calc(inp[i], h)
                    loss += loss_fn(out, target[i])
                
                return loss.unsqueeze(0)

            elif not reward is None:
                _, h = self.calc_cond(x0, h)
                for i in range(seq_len):
                    out, h = self.calc(inp[i], h)
                    for j in range(batch_size):
                        loss += -out[j][target.data[i][j]]*reward[j]

                return loss.unsqueeze(0)/batch_size

    def calc_cond(self, inp, hidden):
        inp = inp.view(1, -1, self.emb_dim)
        out, hidden = self.gru(inp, hidden)
        out = self.gru2out(out.view(-1, self.hid_dim))
        out = F.log_softmax(out, dim=1)

        return out, hidden

    def calc(self, inp, hidden):
        emb = self.embeddings(inp)
        emb = emb.view(1, -1, self.emb_dim)
        out, hidden = self.gru(emb, hidden)
        out = self.gru2out(out.view(-1, self.hid_dim))
        out = F.log_softmax(out, dim=1)

        return out, hidden

    def sample(self, num_samples, cond, start_letter=0):
        samples = torch.zeros(num_samples, self.seq_len).type(torch.LongTensor)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples))

        if self.gpu_num > 0:
            samples = samples.cuda()
            inp = inp.cuda()

        _, h = self.calc_cond(cond, h)
        for i in range(self.seq_len):
            out, h = self.calc(inp, h)
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data

            inp = out.view(-1)

        return samples
