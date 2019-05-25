import chainer
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I
from chainer.backends import cuda

import cupy as xp
import numpy as np

t_dim = 128

class RNN_Encoder(chainer.Chain):
    def __init__(self, n_layer, dropout, embed_mat, wscale=0.02):
        super(RNN_Encoder, self).__init__()
        with self.init_scope():
            voc_dim, emb_dim = embed_mat.shape
            self.embed = L.EmbedID(voc_dim, emb_dim, initialW=embed_mat)
            self.LSTM = L.NStepLSTM(n_layer, emb_dim, t_dim, dropout)
    
    def forward(self, xs):
        xs = [self.embed(_xs) for _xs in xs]
        hs, _, _ = self.LSTM(None, None, xs)

        return hs[-1,:,:]


class CNN_Encoder(chainer.Chain):
    def __init__(self, wscale=0.02):
        df_dim = 64
        super(CNN_Encoder, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0_conv = L.Convolution2D(3, df_dim, ksize=(4,4), stride=(2,2), pad=(1,1), initialW=w, initial_bias=w)
            self.l1_conv = L.Convolution2D(df_dim, df_dim*2, ksize=(4,4), stride=(2,2), pad=(1,1), initialW=w, nobias=True)
            self.l1_bn = L.BatchNormalization(df_dim*2, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=df_dim*2)))

            self.l2_conv = L.Convolution2D(df_dim*2, df_dim*4, ksize=(4,4), stride=(2,2), pad=(1,1), initialW=w, nobias=True)
            self.l2_bn = L.BatchNormalization(df_dim*4, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=df_dim*4)))

            self.l3_conv = L.Convolution2D(df_dim*4, df_dim*8, ksize=(4,4), stride=(2,2), pad=(1,1), initialW=w, nobias=True)
            self.l3_bn = L.BatchNormalization(df_dim*8, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=df_dim*8)))

            self.l4_dense = L.Linear(df_dim*8*4*4, t_dim, initialW=w, nobias=True)

    def forward(self, xs):
        hs_0 = F.leaky_relu(self.l0_conv(xs), slope=0.2)
        hs_1 = F.leaky_relu(self.l1_bn(self.l1_conv(hs_0)), slope=0.2)
        hs_2 = F.leaky_relu(self.l2_bn(self.l2_conv(hs_1)), slope=0.2)
        hs_3 = F.leaky_relu(self.l3_bn(self.l3_conv(hs_2)), slope=0.2)
        hs_4 = self.l4_dense(F.flatten(hs_3).reshape(len(xs), -1))

        return hs_4


class DS_SJE(chainer.Chain):
    def __init__(self, args, embed_mat, alpha=0.2, wscale=0.02):
        self.n_layer = args.n_layer
        self.dropout = args.dropout
        self.alpha = alpha
        self._gpu_id = None
        super(DS_SJE, self).__init__()
        with self.init_scope():
            self.rnn_encoder = RNN_Encoder(self.n_layer, self.dropout, embed_mat)
            self.cnn_encoder = CNN_Encoder()
        
    def forward(self, img, text, w_img, w_text):
        x = self.rnn_encoder(text)
        w_x = self.rnn_encoder(w_text)
        v = self.cnn_encoder(img)
        w_v = self.cnn_encoder(w_img)

        zeros = cuda.to_gpu(xp.array(0., dtype="float32"), self._gpu_id)
        loss = F.mean(F.maximum(zeros, self.alpha - cosine_similarity(x, v) + cosine_similarity(x, w_v))) +\
                    F.mean(F.maximum(zeros, self.alpha - cosine_similarity(x, v) + cosine_similarity(w_x, v)))

        return x, w_x, loss
    
    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, value):
        self._gpu_id = value
    
    @gpu_id.getter
    def gpu_id(self):
        return self._gpu_id

def cosine_similarity(v1, v2):
    cost =  F.matmul(v1,v2.T) / F.sqrt(F.batch_l2_norm_squared(v1) * F.batch_l2_norm_squared(v2))
    return cost