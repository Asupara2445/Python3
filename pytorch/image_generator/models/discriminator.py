import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.network_utils import *

class ImageDown(torch.nn.Module):
    def __init__(self, input_size, num_chan, out_dim):
        super(ImageDown, self).__init__()
        self.__dict__.update(locals())

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        activ = nn.LeakyReLU(0.2, True)

        _layers = []
        if input_size == 64:
            cur_dim = 128
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)]  # 32
            _layers += [conv_norm(cur_dim, cur_dim*2, norm_layer, stride=2, activation=activ)]  # 16
            _layers += [conv_norm(cur_dim*2, cur_dim*4, norm_layer, stride=2, activation=activ)]  # 8
            _layers += [conv_norm(cur_dim*4, out_dim,  norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)]  # 4

        if input_size == 128:
            cur_dim = 64
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)]  # 64
            _layers += [conv_norm(cur_dim, cur_dim*2, norm_layer, stride=2, activation=activ)]  # 32
            _layers += [conv_norm(cur_dim*2, cur_dim*4, norm_layer, stride=2, activation=activ)]  # 16
            _layers += [conv_norm(cur_dim*4, cur_dim*8, norm_layer, stride=2, activation=activ)]  # 8
            _layers += [conv_norm(cur_dim*8, out_dim,  norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)]  # 4

        if input_size == 256:
            cur_dim = 32 # for testing
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)] # 128
            _layers += [conv_norm(cur_dim, cur_dim*2,  norm_layer, stride=2, activation=activ)] # 64
            _layers += [conv_norm(cur_dim*2, cur_dim*4,  norm_layer, stride=2, activation=activ)] # 32
            _layers += [conv_norm(cur_dim*4, cur_dim*8,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*8, out_dim,  norm_layer, stride=2, activation=activ)] # 8

        self.node = nn.Sequential(*_layers)

    def forward(self, inputs):
        out = self.node(inputs)
        return out

class DiscClassifier(nn.Module):
    def __init__(self, enc_dim, emb_dim, kernel_size):
        super(DiscClassifier, self).__init__()
        self.__dict__.update(locals())
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        activ = nn.LeakyReLU(0.2, True)
        inp_dim = enc_dim + emb_dim

        _layers = [conv_norm(inp_dim, enc_dim, norm_layer, kernel_size=1, stride=1, activation=activ),
                   nn.Conv2d(enc_dim, 1, kernel_size=kernel_size, padding=0, bias=True)]

        self.node = nn.Sequential(*_layers)

    def forward(self, sent_code,  img_code):
        sent_code = sent_code.unsqueeze(-1).unsqueeze(-1)
        dst_shape = list(sent_code.size())
        dst_shape[1] = sent_code.size()[1]
        dst_shape[2] = img_code.size()[2]
        dst_shape[3] = img_code.size()[3]
        sent_code = sent_code.expand(dst_shape)
        comp_inp = torch.cat([img_code, sent_code], dim=1)
        output = self.node(comp_inp)
        chn = output.size()[1]
        output = output.view(-1, chn)

        return output

class DISCRIMINATOR(torch.nn.Module):
    def __init__(self, num_chan,  hid_dim, sent_dim, emb_dim, side_output_at=[64, 128, 256]):
        super(DISCRIMINATOR, self).__init__()
        self.__dict__.update(locals())

        activ = nn.LeakyReLU(0.2, True)
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

        self.side_output_at = side_output_at

        enc_dim = hid_dim * 4  # the ImageDown output dimension

        if 64 in side_output_at:  # discriminator for 64 input
            self.img_encoder_64 = ImageDown(64,  num_chan,  enc_dim)  # 4x4
            self.pair_disc_64 = DiscClassifier(enc_dim, emb_dim, kernel_size=4)
            self.local_img_disc_64 = nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_64 = nn.Sequential(*_layers)

        if 128 in side_output_at:  # discriminator for 128 input
            self.img_encoder_128 = ImageDown(128,  num_chan, enc_dim)  # 4x4
            self.pair_disc_128 = DiscClassifier(enc_dim, emb_dim, kernel_size=4)
            self.local_img_disc_128 = nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_128 = nn.Sequential(*_layers)

        if 256 in side_output_at:  # discriminator for 256 input
            self.img_encoder_256 = ImageDown(256, num_chan, enc_dim)     # 8x8
            self.pair_disc_256 = DiscClassifier(enc_dim, emb_dim, kernel_size=4)
            self.pre_encode = conv_norm(enc_dim, enc_dim, norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)
            self.local_img_disc_256 = nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_256 = nn.Sequential(*_layers)

        self.apply(weights_init)

    def forward(self, images, embedding):
        out_dict = OrderedDict()
        this_img_size = images.size()[3]
        assert this_img_size in [32, 64, 128, 256], 'wrong input size {} in image discriminator'.format(this_img_size)

        img_encoder = getattr(self, 'img_encoder_{}'.format(this_img_size))
        local_img_disc = getattr(
            self, 'local_img_disc_{}'.format(this_img_size), None)
        pair_disc = getattr(self, 'pair_disc_{}'.format(this_img_size))
        context_emb_pipe = getattr(
            self, 'context_emb_pipe_{}'.format(this_img_size))

        sent_code = context_emb_pipe(embedding)
        img_code = img_encoder(images)

        if this_img_size == 256:
            pre_img_code = self.pre_encode(img_code)
            pair_disc_out = pair_disc(sent_code, pre_img_code)
        else:
            pair_disc_out = pair_disc(sent_code, img_code)

        local_img_disc_out = local_img_disc(img_code)

        return pair_disc_out, local_img_disc_out