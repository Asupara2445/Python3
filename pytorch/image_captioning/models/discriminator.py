import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd


class DISCRIMINATOR(nn.Module):

    def __init__(self, n_voc, emb_dim, hid_dim, img_dim, seq_len, gpu_num, dropout=0.2):
        super(DISCRIMINATOR, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.img_dim = img_dim
        self.seq_len = seq_len
        self.gpu_num = gpu_num

        self.encoder = nn.Linear(4096, self.img_dim)
        self.embeddings = nn.Embedding(n_voc, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hid_dim, hid_dim)
        self.dropout_linear = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hid_dim))

        if self.gpu_num > 0:
            return h.cuda()
        else:
            return h

    def forward(self, input, target=None):
        sent, img = input
        h = self.init_hidden(sent.size()[0])

        img_feat = self.encoder(img)
        out = self.calc(sent, img_feat, h)

        if not target is None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(out, target)
            acc = torch.sum((out>0.5)==(target>0.5)).float()
            return loss.unsqueeze(0), acc.unsqueeze(0)
        
        else:
            return out.view(-1)

    def calc(self, sent, img_feat, hidden):
        emb = self.embeddings(sent)
        emb = emb.permute(1, 0, 2)
        self.gru.flatten_parameters()
        _, hidden = self.gru(emb, hidden)
        hidden = hidden.permute(1, 0, 2).contiguous()
        out = self.gru2hidden(hidden.view(-1, 4*self.hid_dim))
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = (out*img_feat).sum(1, keepdim=True)
        out = torch.sigmoid(out)
        return out