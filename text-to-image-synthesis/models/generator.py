import chainer
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I
from chainer.backends import cuda

import cupy as xp
import numpy as np

t_dim = 128

class Generator(chainer.Chain):
    def __init__(self, args, wscale=0.02):
        self.sent_dim = args.sent_dim
        self.noise_dim = args.noise_dim
        self.image_size = args.image_size[0]
        self.batch_size = args.batch_size // 2
        self.noise_dist = args.noise_dist
        self._gpu_id = None
        super(Generator, self).__init__()
        with self.init_scope():
            self.s16 = int(self.image_size/16)
            w = I.Normal(wscale)
            gf_dim = 128
            self.l0_dense = L.Linear(t_dim, self.sent_dim, initialW=w, initial_bias=w)
            self.l1_dense = L.Linear(self.sent_dim+self.noise_dim, gf_dim*8*self.s16*self.s16, initialW=w, nobias=True)
            self.l1_bn = L.BatchNormalization(gf_dim*8*self.s16*self.s16, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=gf_dim*8*self.s16*self.s16)))

            self.l2_conv = L.Convolution2D(gf_dim*8, gf_dim*2, ksize=(1,1), stride=(1,1), pad=(0,0), initialW=w, nobias=True)
            self.l2_bn = L.BatchNormalization(gf_dim*2, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=gf_dim*2)))
            self.l3_conv = L.Convolution2D(gf_dim*2, gf_dim*2, ksize=(3,3), stride=(1,1), pad=(1,1), initialW=w, nobias=True)
            self.l3_bn = L.BatchNormalization(gf_dim*2, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=gf_dim*2)))
            self.l4_conv = L.Convolution2D(gf_dim*2, gf_dim*8, ksize=(3,3), stride=(1,1), pad=(1,1), initialW=w, nobias=True)
            self.l4_bn = L.BatchNormalization(gf_dim*8, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=gf_dim*8)))

            self.l5_deconv = L.Deconvolution2D(gf_dim*8, gf_dim*8, ksize=(2,2), stride=(2,2), pad=(0,0), initialW=w, initial_bias=w)
            self.l5_conv = L.Convolution2D(gf_dim*8, gf_dim*4, ksize=(3,3), stride=(1,1), pad=(1,1), initialW=w, nobias=True)
            self.l5_bn = L.BatchNormalization(gf_dim*4, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=gf_dim*4)))

            self.l6_conv = L.Convolution2D(gf_dim*4, gf_dim, ksize=(1,1), stride=(1,1), pad=(0,0), initialW=w, nobias=True)
            self.l6_bn = L.BatchNormalization(gf_dim, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=gf_dim)))
            self.l7_conv = L.Convolution2D(gf_dim, gf_dim, ksize=(3,3), stride=(1,1), pad=(1,1), initialW=w, nobias=True)
            self.l7_bn = L.BatchNormalization(gf_dim, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=gf_dim)))
            self.l8_conv = L.Convolution2D(gf_dim, gf_dim*4, ksize=(3,3), stride=(1,1), pad=(1,1), initialW=w, nobias=True)
            self.l8_bn = L.BatchNormalization(gf_dim*4, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=gf_dim*4)))

            self.l9_deconv = L.Deconvolution2D(gf_dim*4, gf_dim*4, ksize=(2,2), stride=(2,2), pad=(0,0), initialW=w, initial_bias=w)
            self.l9_conv = L.Convolution2D(gf_dim*4, gf_dim*2, ksize=(3,3), stride=(1,1), pad=(1,1), initialW=w, nobias=True)
            self.l9_bn = L.BatchNormalization(gf_dim*2, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=gf_dim*2)))

            self.l10_deconv = L.Deconvolution2D(gf_dim*2, gf_dim*2, ksize=(2,2), stride=(2,2), pad=(0,0), initialW=w, initial_bias=w)
            self.l10_conv = L.Convolution2D(gf_dim*2, gf_dim, ksize=(3,3), stride=(1,1), pad=(1,1), initialW=w, nobias=True)
            self.l10_bn = L.BatchNormalization(gf_dim, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=gf_dim)))

            self.l11_deconv = L.Deconvolution2D(gf_dim, gf_dim, ksize=(2,2), stride=(2,2), pad=(0,0), initialW=w, initial_bias=w)
            self.l11_conv = L.Convolution2D(gf_dim, 3, ksize=(3,3), stride=(1,1), pad=(1,1), initialW=w, initial_bias=w)
    
    def make_noise(self, size=None):
        if size is None:size = self.batch_size
        if self.noise_dist == "normal":
            array =  xp.random.randn(size, self.noise_dim).astype(xp.float32)
        elif self.noise_dist == "uniform":
            array =  xp.random.uniform(-1, 1, (size, self.noise_dim)).astype(xp.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.noise_dist)
        
        if not self._gpu_id is None:array = cuda.to_gpu(array, self._gpu_id)
        return chainer.Variable(array)

    def forward(self, xs):
        zs =self.make_noise()

        hs_0 = F.leaky_relu(self.l0_dense(xs), slope=0.2)
        hs_0 = F.concat([hs_0, zs], axis=1)
        hs_0 = self.l1_bn(self.l1_dense(hs_0)).reshape((self.batch_size, -1, self.s16, self.s16))

        hs_1 = F.relu(self.l2_bn(self.l2_conv(hs_0)))
        hs_1 = F.relu(self.l3_bn(self.l3_conv(hs_1)))
        hs_1 = self.l4_bn(self.l4_conv(hs_1))

        hs_2 = F.relu(F.add(hs_0, hs_1))
        hs_2 = self.l5_deconv(hs_2)
        hs_2 = self.l5_bn(self.l5_conv(hs_2))

        hs_3 = F.relu(self.l6_bn(self.l6_conv(hs_2)))
        hs_3 = F.relu(self.l7_bn(self.l7_conv(hs_3)))
        hs_3 = self.l8_bn(self.l8_conv(hs_3))

        hs_4 = F.relu(F.add(hs_2, hs_3))
        hs_4 = self.l9_deconv(hs_4)
        hs_4 = F.relu(self.l9_bn(self.l9_conv(hs_4)))

        hs_5 = self.l10_deconv(hs_4)
        hs_5 = F.relu(self.l10_bn(self.l10_conv(hs_5)))

        hs_6 = self.l11_deconv(hs_5)
        hs_6 = F.tanh(self.l11_conv(hs_6))

        return hs_6
    
    def generate(self, xs):
        zs =self.make_noise(len(xs))

        hs_0 = F.leaky_relu(self.l0_dense(xs), slope=0.2)
        hs_0 = F.concat([hs_0, zs], axis=1)
        hs_0 = self.l1_bn(self.l1_dense(hs_0)).reshape((len(xs), -1, self.s16, self.s16))

        hs_1 = F.relu(self.l2_bn(self.l2_conv(hs_0)))
        hs_1 = F.relu(self.l3_bn(self.l3_conv(hs_1)))
        hs_1 = self.l4_bn(self.l4_conv(hs_1))

        hs_2 = F.relu(F.add(hs_0, hs_1))
        hs_2 = self.l5_deconv(hs_2)
        hs_2 = self.l5_bn(self.l5_conv(hs_2))

        hs_3 = F.relu(self.l6_bn(self.l6_conv(hs_2)))
        hs_3 = F.relu(self.l7_bn(self.l7_conv(hs_3)))
        hs_3 = self.l8_bn(self.l8_conv(hs_3))

        hs_4 = F.relu(F.add(hs_2, hs_3))
        hs_4 = self.l9_deconv(hs_4)
        hs_4 = F.relu(self.l9_bn(self.l9_conv(hs_4)))

        hs_5 = self.l10_deconv(hs_4)
        hs_5 = F.relu(self.l10_bn(self.l10_conv(hs_5)))

        hs_6 = self.l11_deconv(hs_5)
        hs_6 = F.tanh(self.l11_conv(hs_6))

        return cuda.to_cpu(hs_6.data)

    
    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, value):
        self._gpu_id = value
    
    @gpu_id.getter
    def gpu_id(self):
        return self._gpu_id