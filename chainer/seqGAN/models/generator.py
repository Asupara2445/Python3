import chainer
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I
from chainer.backends import cuda

import numpy as np
import cupy as xp


class Generator(chainer.Chain):
    def __init__(self, n_voc, emb_dim, hid_dim, seq_len, gpu_num):
        super(Generator, self).__init__()
        self.n_voc = n_voc
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.gpu_num = gpu_num
        self._gpu_id = None
        self.voc_size = xp.array([i for i in range(n_voc)])

        w = I.Normal(1.)
        with self.init_scope():
            self.embeddings = L.EmbedID(n_voc, emb_dim, initialW=w)
            self.gru = L.GRU(emb_dim, hid_dim, init=w, inner_init=w, bias_init=I.Zero())
            self.gru2out = L.Linear(hid_dim, n_voc, initialW=w, initial_bias=I.Zero())

    def forward(self, inp, target=None, reward=None):
        self.gru.reset_state()
        if target is None and reward is None:
            sample_size = inp
            return self.sample(sample_size)

        else:
            batch_size, seq_len = inp.shape
            inp = inp.T
            target = target.T

            loss = 0
            if not target is None and reward is None:
                for i in range(seq_len):
                    out = self.calc(inp[i])
                    loss += F.bernoulli_nll(out, target[i])

                return loss

            elif not target is None and not reward is None:
                for i in range(seq_len):
                    out = self.calc(inp[i])
                    for j in range(batch_size):
                        loss += -out[j][target[i][j]]*reward[j]

                return loss / batch_size

    def calc(self, inp):
        emb = self.embeddings(inp)
        out = self.gru(emb)
        out = self.gru2out(out)
        out = F.log_softmax(out, axis=1)

        return out

    def sample(self, num_samples, start_letter=0):
        inp = xp.array([start_letter]*num_samples)
        samples = xp.zeros((self.seq_len, num_samples))

        if self.gpu_num > 0:
            inp = cuda.to_gpu(inp, self._gpu_id)

        for i in range(self.seq_len):
            out = self.calc(inp)
            out = F.exp(out)
            out = [xp.random.choice(self.voc_size, size=1, replace=True, p=pred) for pred in out.data]
            out = xp.concatenate(out)
            samples[i] = out

            if self.gpu_num > 0:
                out = cuda.to_gpu(out, self._gpu_id)
            inp = out.reshape(-1)

        return samples.T
    
    @property
    def gpu_id(self):
        return self._gpu_id
    
    @gpu_id.setter
    def gpu_id(self, id):
        self._gpu_id = id