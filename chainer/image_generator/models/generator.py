import chainer
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I
from chainer.backends import cuda

import numpy as np
import cupy as xp

img_scales = [64, 128, 256]
num_scales = [4, 8, 16, 32, 64, 128]
reduce_dim_at = [8, 32, 128, 256]

class pad_conv_norm(chainer.Chain):
    def __init__(self, dim_in, dim_out, kernel_size=3, activ=F.relu, wscale=0.02):
        super(pad_conv_norm, self).__init__()
        self.activ = activ
        w = I.Normal(wscale)
        with self.init_scope():
            if kernel_size == 1:
                self.conv = L.Convolution2D(dim_in, dim_out, ksize=kernel_size, stride=1, pad=0, nobias=True, initialW=w)
            else:
                self.conv = L.Convolution2D(dim_in, dim_out, ksize=kernel_size, stride=1, pad=1, nobias=True, initialW=w)

            self.bn = L.BatchNormalization(dim_out, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=dim_out)))

    def forward(self, xs):
        hs = self.bn(self.conv(xs))

        if self.activ is not None:
            hs = self.activ(hs)
        
        return hs

class ResnetBlock(chainer.Chain):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        with self.init_scope():
            self.conv_0 = pad_conv_norm(dim, dim)
            self.conv_1 = pad_conv_norm(dim, dim, activ=None)
    
    def forward(self, xs):
        hs = self.conv_0(xs)
        hs = self.conv_1(hs)

        return hs + xs

class condEmbedding(chainer.Chain):
    def __init__(self, sen_dim, emb_dim, wscale=0.02):
        super(condEmbedding, self).__init__()
        w = I.Normal(wscale)
        with self.init_scope():
            self.l1_mu = L.Linear(sen_dim, emb_dim, initialW=w, initial_bias=w)
            self.l1_ln_var = L.Linear(sen_dim, emb_dim, initialW=w, initial_bias=w)
    
    def forward(self, hs):
        data_len = len(hs)

        mu = self.l1_mu(hs)
        ln_var = self.l1_ln_var(hs)

        mu = F.leaky_relu(mu, slope=0.2)
        ln_var = F.leaky_relu(ln_var, slope=0.2)

        zs = F.gaussian(mu, ln_var)
        loss = F.gaussian_kl_divergence(mu, ln_var) / data_len
    
        return zs, loss

class Up_Sapmling(chainer.Chain):
    def __init__(self, dim_in, dim_out, kernel_size=4, stride=2, pad=1, use_bias=True, 
                 use_activ=True, use_bn=True, activ=F.relu, w=None):
        super(Up_Sapmling, self).__init__()
        self.use_bn = use_bn
        self.use_activ = use_activ
        self.activ = activ
        with self.init_scope():
            self.deconv =L.Deconvolution2D(dim_in, dim_out, ksize=kernel_size,
                                            stride=stride, pad=pad, nobias=use_bias, initialW=w)
            if self.use_bn:
                self.bn = L.BatchNormalization(dim_out)

    def forward(self, xs):
        hs = self.deconv(xs)
        if self.use_bn:
            hs = self.bn(hs)
        if self.use_activ:
            hs = self.activ(hs)

        return hs

class Sent2FeatMap(chainer.Chain):
    def __init__(self, sent_dim, noise_dim, row, col, channel, wscale=0.02, activ=None):
        super(Sent2FeatMap, self).__init__()
        self.row = row
        self.col = col
        self.channel = channel
        self.activ = activ
        w = I.Normal(wscale)
        with self.init_scope():
            out_dim = row*col*channel
            self.l1_dense = L.Linear(sent_dim+noise_dim, out_dim, nobias=True, initialW=w)
            self.l1_bn = L.BatchNormalization(out_dim, initial_gamma=I.Constant(np.random.normal(loc=1., scale=wscale, size=out_dim)))
    
    def forward(self, hs):
        hs = self.l1_bn(self.l1_dense(hs))
        if self.activ is not None:
            hs = self.activ(hs)
        
        hs = hs.reshape((-1, self.channel, self.row, self.col))

        return hs

class Scale_Block(chainer.Chain):
    def __init__(self, scale, cur_dim, num_resblock, wscale=0.02):
        super(Scale_Block, self).__init__()
        self.scale = scale
        self.num_resblock = num_resblock
        
        w = I.Normal(wscale)
        with self.init_scope():
            if self.scale != 4:
                self.upsample = Up_Sapmling(cur_dim, cur_dim)

            if self.scale in reduce_dim_at:
                self.red_conv = pad_conv_norm(cur_dim, cur_dim//2)
                cur_dim = cur_dim // 2
            
            for i in range(self.num_resblock):
                setattr(self, f"resblock_{i}", ResnetBlock(cur_dim))
            
            if self.scale in img_scales:
                self.tensor2img = L.Convolution2D(cur_dim, 3, ksize=3, stride=1, pad=1, initialW=w, initial_bias=w)

    def forward(self, hs):
        if self.scale != 4:
            hs = self.upsample(hs)
        if self.scale in reduce_dim_at:
            hs = self.red_conv(hs)
        
        for i in range(self.num_resblock):
            res_block = getattr(self, f"resblock_{i}")
            hs = res_block(hs)
        
        if self.scale in img_scales:
            img = self.tensor2img(hs)
            img = F.tanh(img)

            return hs, img

        return hs


class Generator(chainer.Chain):
    def __init__(self, args, wscale=0.02):
        super(Generator, self).__init__()
        self._gpu_id = None
        self.hid_dim = args.hid_dim
        self.emb_dim = args.emb_dim
        self.sent_dim = args.sent_dim
        self.noise_dim = args.noise_dim
        self.batch_size = args.batch_size // args.gpu_num
        self.noise_dist = args.noise_dist
        self.num_resblock = args.num_resblock

        with self.init_scope():
            self.condEmbedding = condEmbedding(self.sent_dim, self.emb_dim)

            self.vector2tensor = Sent2FeatMap(self.emb_dim, self.noise_dim, 4, 4, self.hid_dim*8)

            cur_dim = self.hid_dim*8
            for i in range(len(num_scales)):
                setattr(self, f"fg_scale_{num_scales[i]}", Scale_Block(num_scales[i], cur_dim, self.num_resblock))
                setattr(self, f"bg_scale_{num_scales[i]}", Scale_Block(num_scales[i], cur_dim, self.num_resblock))
                if num_scales[i] in reduce_dim_at:
                    cur_dim = cur_dim // 2
            
            self.scale_128 = Scale_Block(128, cur_dim*4, self.num_resblock)
            self.scale_256 = Scale_Block(256, cur_dim*4, self.num_resblock)
    
    def make_noise(self):
        if self.noise_dist == "normal":
            array =  xp.random.randn(self.batch_size, self.noise_dim).astype(xp.float32)
        elif self.noise_dist == "uniform":
            array =  xp.random.uniform(-1, 1, (self.batch_size, self.noise_dim)).astype(xp.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.noise_dist)
        
        if not self._gpu_id is None:array = cuda.to_gpu(array, self._gpu_id)
        return chainer.Variable(array)

    def forward(self, xs):
        zs = self.make_noise()
        hs, KL_loss = self.condEmbedding(xs)
        hs = F.concat([hs, zs], axis=1)
        hs = self.vector2tensor(hs)
        
        bg_x_4 = self.bg_scale_4(hs)
        bg_x_8 = self.bg_scale_8(bg_x_4)
        bg_x_16 = self.bg_scale_16(bg_x_8)
        bg_x_32 = self.bg_scale_32(bg_x_16)
        bg_x_64, bg_img_64 = self.bg_scale_64(bg_x_32)
        bg_x_128, bg_img_128 = self.bg_scale_128(bg_x_64)

        fg_x_4 = self.fg_scale_4(hs)
        fg_x_8 = self.fg_scale_8(fg_x_4)
        fg_x_16 = self.fg_scale_16(fg_x_8)
        fg_x_32 = self.fg_scale_32(fg_x_16)
        fg_x_64, fg_img_64 = self.fg_scale_64(fg_x_32)
        fg_x_128, fg_img_128 = self.fg_scale_128(fg_x_64)

        x_64 = F.concat([bg_x_64, fg_x_64], axis=1)
        x_128, img_128 = self.scale_128(x_64)

        x_128 = F.concat([bg_x_128, fg_x_128, x_128], axis=1)
        _, img_256 = self.scale_256(x_128)

        return {"bg_img_64":bg_img_64,"bg_img_128":bg_img_128,"fg_img_64":fg_img_64,"fg_img_128":fg_img_128,"x_128":img_128,"x_256":img_256}, KL_loss

    def generate(self, xs, zs):
        zs = self.make_noise()
        hs, _ = self.condEmbedding(xs)
        hs = F.concat([hs, zs], axis=1)
        hs = self.vector2tensor(hs)
        
        x_4 = self.scale_4(hs)
        x_8 = self.scale_8(x_4)
        x_16 = self.scale_16(x_8)
        x_32 = self.scale_32(x_16)

        x_64, img_64 = self.scale_64(x_32)
        x_128, img_128 = self.scale_128(x_64)
        _, img_256 = self.scale_256(x_128)

        return {"x_64":img_64,"x_128":img_128,"x_256":img_256}
    
    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, value):
        self._gpu_id = value
    
    @gpu_id.getter
    def gpu_id(self):
        return self._gpu_id
