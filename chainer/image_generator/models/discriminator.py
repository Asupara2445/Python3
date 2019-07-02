import chainer
import chainer.links as L
import chainer.functions as F
from chainer.backends import cuda

import cupy as xp

class conv_norm(chainer.Chain):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, activation=F.relu,
                            use_bias=False, use_norm=True, padding=None, wscale=0.02):
        super(conv_norm, self).__init__()
        self.activation = activation
        self.use_norm = use_norm
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            if kernel_size == 3:
                padding = 1 if padding is None else padding
            else:
                padding = 0 if padding is None else padding
            
            self.conv = L.Convolution2D(dim_in, dim_out, ksize=kernel_size, stride=stride, pad=padding, nobias=use_bias, initialW=w)

            if self.use_norm:self.bn = L.BatchNormalization(dim_out)

    def forward(self, xs):
        hs = self.conv(xs)
        if self.use_norm:hs = self.bn(hs)
        if self.activation is not None:
            hs = self.activation(hs)
        
        return hs

class condEmbedding(chainer.Chain):
    def __init__(self, sent_dim, out_dim, initW=None):
        self.sent_dim = sent_dim
        self.out_dim = out_dim
        super(condEmbedding, self).__init__()
        with self.init_scope():
            self.l2 = L.Linear(self.sent_dim, self.out_dim, initialW=initW)
    
    def forward(self, hs):
        hs = self.l2(hs)
        hs = F.leaky_relu(hs, slope=0.2)

        return hs

class ImageDown(chainer.Chain):
    def __init__(self, image_size, num_chan, out_dim):
        super(ImageDown, self).__init__()
        self.image_size = image_size
        with self.init_scope():
            if image_size == 64:
                cur_dim = 128
                self.l1_conv = conv_norm(num_chan, cur_dim, stride=2, activation=F.leaky_relu, use_norm=False)
                self.l2_conv = conv_norm(cur_dim, cur_dim*2, stride=2, activation=F.leaky_relu)
                self.l3_conv = conv_norm(cur_dim*2, cur_dim*4, stride=2, activation=F.leaky_relu)
                self.l4_conv = conv_norm(cur_dim*4, out_dim, stride=1, kernel_size=5, padding=0, activation=F.leaky_relu)
            
            elif image_size == 128:
                cur_dim = 64
                self.l1_conv = conv_norm(num_chan, cur_dim, stride=2, activation=F.leaky_relu, use_norm=False)
                self.l2_conv = conv_norm(cur_dim, cur_dim*2, stride=2, activation=F.leaky_relu)
                self.l3_conv = conv_norm(cur_dim*2, cur_dim*4, stride=2, activation=F.leaky_relu)
                self.l4_conv = conv_norm(cur_dim*4, cur_dim*8, stride=2, activation=F.leaky_relu)
                self.l5_conv = conv_norm(cur_dim*8, out_dim, stride=1, kernel_size=5, padding=0, activation=F.leaky_relu)

            elif image_size == 256:
                cur_dim = 32
                self.l1_conv = conv_norm(num_chan, cur_dim, stride=2, activation=F.leaky_relu, use_norm=False)
                self.l2_conv = conv_norm(cur_dim, cur_dim*2, stride=2, activation=F.leaky_relu)
                self.l3_conv = conv_norm(cur_dim*2, cur_dim*4, stride=2, activation=F.leaky_relu)
                self.l4_conv = conv_norm(cur_dim*4, cur_dim*8, stride=2, activation=F.leaky_relu)
                self.l5_conv = conv_norm(cur_dim*8, out_dim, stride=2, activation=F.leaky_relu)

    def forward(self, xs):
        hs = self.l1_conv(xs)
        hs = self.l2_conv(hs)
        hs = self.l3_conv(hs)
        hs = self.l4_conv(hs)

        if self.image_size != 64:
            hs = self.l5_conv(hs)
        
        return hs

class DiscClassifier(chainer.Chain):
    def __init__(self, enc_dim, sent_dim, kernel_size):
        super(DiscClassifier, self).__init__()
        with self.init_scope():
            self.conv_0 = conv_norm(enc_dim+sent_dim, enc_dim, kernel_size=1, stride=1, activation=F.leaky_relu)
            self.conv_1 = L.Convolution2D(enc_dim, 1, kernel_size, pad=0)
    
    def forward(self, cs, hs):
        cs = cs.reshape((cs.shape[0],cs.shape[1],1,1))
        cs = F.tile(cs, (1, 1, 4, 4))
        hs = F.concat([hs,cs], axis=1)
        hs = self.conv_0(hs)
        hs = self.conv_1(hs).reshape((-1,1))

        return hs


class Discriminator(chainer.Chain):
    def __init__(self, args, num_chan=3, num_resblock=1, wscale=0.02, side_output_at=[64, 128, 256]):
        self.sent_dim = args.sent_dim
        self.hid_dim = args.hid_dim
        self.emb_dim = args.emb_dim
        self.side_output_at = side_output_at

        enc_dim = self.hid_dim * 4
        super(Discriminator, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            if 64 in self.side_output_at:
                self.fg_img_encoder_64 = ImageDown(64,  num_chan,  enc_dim)
                self.fg_pair_disc_64 = DiscClassifier(enc_dim, self.emb_dim, kernel_size=4)
                self.fg_local_img_disc_64 = L.Convolution2D(enc_dim, 1, 4, pad=0, initialW=w, initial_bias=w)
                self.fg_context_emb_pipe_64 = L.Linear(self.sent_dim, self.emb_dim, initialW=w, initial_bias=w)

                self.bg_img_encoder_64 = ImageDown(64,  num_chan,  enc_dim)
                self.bg_pair_disc_64 = DiscClassifier(enc_dim, self.emb_dim, kernel_size=4)
                self.bg_local_img_disc_64 = L.Convolution2D(enc_dim, 1, 4, pad=0, initialW=w, initial_bias=w)
                self.bg_context_emb_pipe_64 = L.Linear(self.sent_dim, self.emb_dim, initialW=w, initial_bias=w)
            
            if 128 in self.side_output_at:
                self.fg_img_encoder_128 = ImageDown(128,  num_chan,  enc_dim)
                self.fg_pair_disc_128 = DiscClassifier(enc_dim, self.emb_dim, kernel_size=4)
                self.fg_local_img_disc_128 = L.Convolution2D(enc_dim, 1, 4, pad=0, initialW=w, initial_bias=w)
                self.fg_context_emb_pipe_128 = L.Linear(self.sent_dim, self.emb_dim, initialW=w, initial_bias=w)

                self.bg_img_encoder_128 = ImageDown(128,  num_chan,  enc_dim)
                self.bg_pair_disc_128 = DiscClassifier(enc_dim, self.emb_dim, kernel_size=4)
                self.bg_local_img_disc_128 = L.Convolution2D(enc_dim, 1, 4, pad=0, initialW=w, initial_bias=w)
                self.bg_context_emb_pipe_128 = L.Linear(self.sent_dim, self.emb_dim, initialW=w, initial_bias=w)

                self.img_encoder_128 = ImageDown(128,  num_chan,  enc_dim)
                self.pair_disc_128 = DiscClassifier(enc_dim, self.emb_dim, kernel_size=4)
                self.local_img_disc_128 = L.Convolution2D(enc_dim, 1, 4, pad=0, initialW=w, initial_bias=w)
                self.context_emb_pipe_128 = L.Linear(self.sent_dim, self.emb_dim, initialW=w, initial_bias=w)
            
            if 256 in self.side_output_at:
                self.img_encoder_256 = ImageDown(256,  num_chan,  enc_dim)
                self.pair_disc_256 = DiscClassifier(enc_dim, self.emb_dim, kernel_size=4)
                self.pre_encode = conv_norm(enc_dim, enc_dim, stride=1, activation=F.leaky_relu, kernel_size=5, padding=0)
                self.local_img_disc_256 = L.Convolution2D(enc_dim, 1, 4, pad=0, initialW=w, initial_bias=w)
                self.context_emb_pipe_256 = L.Linear(self.sent_dim, self.emb_dim, initialW=w, initial_bias=w)

    def forward(self, images, embedding, fg=False, bg=False):
        this_img_size = images.shape[3]

        if not fg and not bg:
            img_encoder = getattr(self, f"img_encoder_{this_img_size}")
            local_img_disc = getattr(self, f"local_img_disc_{this_img_size}", None)
            pair_disc = getattr(self, f"pair_disc_{this_img_size}")
            context_emb_pipe = getattr(self, f"context_emb_pipe_{this_img_size}")

        elif fg and not bg:
            img_encoder = getattr(self, f"fg_img_encoder_{this_img_size}")
            local_img_disc = getattr(self, f"fg_local_img_disc_{this_img_size}", None)
            pair_disc = getattr(self, f"fg_pair_disc_{this_img_size}")
            context_emb_pipe = getattr(self, f"fg_context_emb_pipe_{this_img_size}")

        elif not fg and bg:
            img_encoder = getattr(self, f"bg_img_encoder_{this_img_size}")
            local_img_disc = getattr(self, f"bg_local_img_disc_{this_img_size}", None)
            pair_disc = getattr(self, f"bg_pair_disc_{this_img_size}")
            context_emb_pipe = getattr(self, f"bg_context_emb_pipe_{this_img_size}")

        sent_code = context_emb_pipe(embedding)
        img_code = img_encoder(images)

        if this_img_size == 256:
            pre_img_code = self.pre_encode(img_code)
            pair_disc_out = pair_disc(sent_code, pre_img_code)
        else:
            pair_disc_out = pair_disc(sent_code, img_code)

        local_img_disc_out = local_img_disc(img_code)

        return pair_disc_out, local_img_disc_out