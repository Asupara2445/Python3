import chainer
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I
from chainer.backends import cuda

import cupy as xp


class Discriminator(chainer.Chain):
    def __init__(self, n_voc, emb_dim, hid_dim, seq_len, gpu_num, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.gpu_num = gpu_num
        self.dropout = dropout

        w = I.Normal(1.)
        with self.init_scope():
            self.embeddings = L.EmbedID(n_voc, emb_dim, initialW=w)
            self.gru = L.NStepBiGRU(2, emb_dim, hid_dim, dropout)
            self.gru2hidden = L.Linear(2*2*hid_dim, hid_dim, initialW=w, initial_bias=I.Zero())
            self.dropout_linear = F.dropout
            self.hidden2out = L.Linear(hid_dim, 1, initialW=w, initial_bias=I.Zero())
    
    def forward(self, inp, target=None):
        out = self.calc(inp)

        if not target is None:
            loss = F.sigmoid_cross_entropy(out, target)
            acc = F.binary_accuracy(out, target)
            
            return loss, acc
        
        else:
            out = F.sigmoid(out)
            return out.reshape(-1)

    def calc(self, inp):
        batch_size = inp.shape[0]
        emb = [self.embeddings(_inp) for _inp in inp]
        ys, _ = self.gru(None, emb)
        ys = ys.transpose(1,0,2)
        out = self.gru2hidden(ys.reshape((-1, 4*self.hid_dim)))
        out = F.tanh(out)
        out = F.dropout(out,  self.dropout)
        out = self.hidden2out(out)
        return out