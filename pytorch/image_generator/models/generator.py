import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.network_utils import *

class condEmbedding(nn.Module):
    def __init__(self, noise_dim, emb_dim):
        super(condEmbedding, self).__init__()

        self.noise_dim = noise_dim
        self.emb_dim = emb_dim
        self.linear  = nn.Linear(noise_dim, emb_dim*2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def sample_encoded_context(self, mean, logsigma, kl_loss=False):
    
        epsilon = Variable(torch.cuda.FloatTensor(mean.size()).normal_())
        stddev  = logsigma.exp()
        
        return epsilon.mul(stddev).add_(mean)

    def forward(self, inputs, kl_loss=True):
        out = self.relu(self.linear(inputs))
        mean = out[:, :self.emb_dim]
        log_sigma = out[:, self.emb_dim:]

        c = self.sample_encoded_context(mean, log_sigma)
        return c, mean, log_sigma

class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        activ = nn.ReLU(True)

        seq = [pad_conv_norm(dim, dim, norm_layer, use_bias=use_bias, activation=activ),
               pad_conv_norm(dim, dim, norm_layer, use_activation=False, use_bias=use_bias)]
        self.res_block = nn.Sequential(*seq)

    def forward(self, input):
        return self.res_block(input) + input

class Sent2FeatMap(nn.Module):
    def __init__(self, in_dim, row, col, channel, activ=None):
        super(Sent2FeatMap, self).__init__()
        self.__dict__.update(locals())
        out_dim = row*col*channel
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True)
        _layers = [nn.Linear(in_dim, out_dim)]
        _layers += [norm_layer(out_dim)]
        if activ is not None:
            _layers += [activ]
        self.out = nn.Sequential(*_layers)

    def forward(self, inputs):
        output = self.out(inputs)
        output = output.view(-1, self.channel, self.row, self.col)
        return output

class GENERATOR(nn.Module):
    def __init__(self, sent_dim, noise_dim, emb_dim, hid_dim, num_resblock=1, side_output_at=[64, 128, 256]):
        super(GENERATOR, self).__init__()
        self.__dict__.update(locals())

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        act_layer = nn.ReLU(True)
        self.condEmbedding = condEmbedding(sent_dim, emb_dim)

        self.vec_to_tensor = Sent2FeatMap(
            emb_dim+noise_dim, 4, 4, self.hid_dim*8)
        self.side_output_at = side_output_at

        reduce_dim_at = [8, 32, 128, 256]
        num_scales = [4, 8, 16, 32, 64, 128, 256]

        cur_dim = self.hid_dim*8
        for i in range(len(num_scales)):
            seq = []
            if i != 0:
                seq += [nn.Upsample(scale_factor=2, mode='nearest')]
            if num_scales[i] in reduce_dim_at:
                seq += [pad_conv_norm(cur_dim, cur_dim//2,
                                      norm_layer, activation=act_layer)]
                cur_dim = cur_dim//2
            for n in range(num_resblock):
                seq += [ResnetBlock(cur_dim)]
            setattr(self, 'scale_%d' % (num_scales[i]), nn.Sequential(*seq))

            if num_scales[i] in self.side_output_at:
                setattr(self, 'tensor_to_img_%d' %
                        (num_scales[i]), branch_out(cur_dim))

        self.apply(weights_init)

    def forward(self, sent_embeddings, z):
        sent_random, mean, logsigma=self.condEmbedding(sent_embeddings)
        
        text = torch.cat([sent_random, z], dim=1)
        x = self.vec_to_tensor(text)
        x_4 = self.scale_4(x)
        x_8 = self.scale_8(x_4)
        x_16 = self.scale_16(x_8)
        x_32 = self.scale_32(x_16)

        x_64 = self.scale_64(x_32)
        output_64 = self.tensor_to_img_64(x_64)

        if 128 in self.side_output_at:
            x_128 = self.scale_128(x_64)
            output_128 = self.tensor_to_img_128(x_128)

            if 256 in self.side_output_at:
                out_256 = self.scale_256(x_128)
                self.keep_out_256 = out_256
                output_256 = self.tensor_to_img_256(out_256)

                return output_64, output_128, output_256, mean, logsigma

            else:
                return output_64, output_128, mean, logsigma

        else:

            return output_64, mean, logsigma