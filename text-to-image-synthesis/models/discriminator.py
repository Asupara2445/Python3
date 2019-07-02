import chainer
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I
from chainer.backends import cuda

import cupy as xp
import numpy as np

t_dim = 128

class Discriminator(chainer.Chain):
    def __init__(self, args, wscale=0.02):
        self.sent_dim = args.sent_dim
        self.image_size = args.image_size[0]
        super(Discriminator, self).__init__()
        df_dim = 64
        with self.init_scope():
            self.s16 = int(self.image_size/16)
            w = chainer.initializers.Normal(wscale)
            self.l0_conv = L.Convolution2D(3, df_dim, ksize=(4,4), stride=(2,2), pad=(1,1), initialW=w, initial_bias=w)

            self.l1_conv = L.Convolution2D(df_dim, df_dim*2, ksize=(4,4), stride=(2,2), pad=(1,1), initialW=w, nobias=True)
            self.l1_bn = L.BatchNormalization(df_dim*2, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=df_dim*2)))
            self.l2_conv = L.Convolution2D(df_dim*2, df_dim*4, ksize=(4,4), stride=(2,2), pad=(1,1), initialW=w, nobias=True)
            self.l2_bn = L.BatchNormalization(df_dim*4, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=df_dim*4)))
            self.l3_conv = L.Convolution2D(df_dim*4, df_dim*8, ksize=(4,4), stride=(2,2), pad=(1,1), initialW=w, nobias=True)
            self.l3_bn = L.BatchNormalization(df_dim*8, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=df_dim*8)))

            self.l4_conv = L.Convolution2D(df_dim*8, df_dim*2, ksize=(1,1), stride=(1,1), pad=(0,0), initialW=w, nobias=True)
            self.l4_bn = L.BatchNormalization(df_dim*2, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=df_dim*2)))
            self.l5_conv = L.Convolution2D(df_dim*2, df_dim*2, ksize=(3,3), stride=(1,1), pad=(1,1), initialW=w, nobias=True)
            self.l5_bn = L.BatchNormalization(df_dim*2, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=df_dim*2)))
            self.l6_conv = L.Convolution2D(df_dim*2, df_dim*8, ksize=(3,3), stride=(1,1), pad=(1,1), initialW=w, nobias=True)
            self.l6_bn = L.BatchNormalization(df_dim*8, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=df_dim*8)))

            self.l7_dense = L.Linear(t_dim, self.sent_dim, initialW=w, initial_bias=w)
            self.l8_conv = L.Convolution2D(self.sent_dim+df_dim*8, df_dim*8, ksize=(1,1), stride=(1,1), pad=(0,0), initialW=w, nobias=True)
            self.l8_bn = L.BatchNormalization(df_dim*8, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=df_dim*8)))

            self.l9_conv = L.Convolution2D(df_dim*8, 1, ksize=(self.s16,self.s16), stride=(self.s16,self.s16), pad=(0,0), initialW=w, initial_bias=w)

    def forward(self, xs, sent_vec):
        hs_0 = F.leaky_relu(self.l0_conv(xs), slope=0.2)

        hs_1 = F.leaky_relu(self.l1_bn(self.l1_conv(hs_0)), slope=0.2)
        hs_1 = F.leaky_relu(self.l2_bn(self.l2_conv(hs_1)), slope=0.2)
        hs_1 = self.l3_bn(self.l3_conv(hs_1))

        hs_2 = F.leaky_relu(self.l4_bn(self.l4_conv(hs_1)), slope=0.2)
        hs_2 = F.leaky_relu(self.l5_bn(self.l5_conv(hs_2)), slope=0.2)
        hs_2 = self.l6_bn(self.l6_conv(hs_2))
        hs_3 = F.leaky_relu(F.add(hs_1, hs_2), slope=0.2)

        sent_vec = F.leaky_relu(self.l7_dense(sent_vec), slope=0.2)
        sent_vec = F.expand_dims(F.expand_dims(sent_vec, axis=2), axis=3)
        sent_vec = F.tile(sent_vec, (1,1,4,4))

        hs_4 = F.concat([hs_3, sent_vec], axis=1)
        hs_4 = F.leaky_relu(self.l8_bn(self.l8_conv(hs_4)), slope=0.2)
        hs_4 = self.l9_conv(hs_4)

        return hs_4.reshape((-1))