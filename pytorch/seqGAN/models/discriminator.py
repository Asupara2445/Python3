import torch
import torch.autograd as autograd
import torch.nn as nn

class DISCRIMINATOR(nn.Module):

    def __init__(self, n_voc, emb_dim, hid_dim, seq_len, gpu_num, dropout=0.2):
        super(DISCRIMINATOR, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.gpu_num = gpu_num

        self.embeddings = nn.Embedding(n_voc, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hid_dim, hid_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hid_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hid_dim))

        if self.gpu_num > 0:
            return h.cuda()
        else:
            return h

    def forward(self, input, target=None):
        h = self.init_hidden(input.size()[0])
        out = self.calc(input, h)

        if not target is None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(out, target)
            acc = torch.sum((out>0.5)==(target>0.5)).float()
            return loss.unsqueeze(0), acc.unsqueeze(0)
        
        else:
            return out.view(-1)

    def calc(self, input, hidden):
        emb = self.embeddings(input)
        emb = emb.permute(1, 0, 2)
        self.gru.flatten_parameters()
        _, hidden = self.gru(emb, hidden)
        hidden = hidden.permute(1, 0, 2).contiguous()
        out = self.gru2hidden(hidden.view(-1, 4*self.hid_dim))
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)
        out = torch.sigmoid(out)
        return out