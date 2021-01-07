import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import utils.logger as logger

class DownsamplingEncoder(nn.Module):
    """VQ-VAE encoder.

    Todo: Check receptive field
    """
    def __init__(self, channels, layer_specs):
        super().__init__()

        self.convs_wide = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.layer_specs = layer_specs
        prev_channels = 1
        total_scale = 1
        pad_left = 0
        self.skips = []

        for stride, ksz, dilation_factor in layer_specs:
            # single loop = single layer

            # --downsample--
            conv_wide = nn.Conv1d(prev_channels, 2 * channels, ksz, stride=stride, dilation=dilation_factor)
            # Initialize weights
            wsize = 2.967 / math.sqrt(ksz * prev_channels)
            conv_wide.weight.data.uniform_(-wsize, wsize)
            conv_wide.bias.data.zero_()
            # Register
            self.convs_wide.append(conv_wide)

            # --1x1 Conv--
            conv_1x1 = nn.Conv1d(channels, channels, 1)
            # Initialize weights
            conv_1x1.bias.data.zero_()
            # Register
            self.convs_1x1.append(conv_1x1)

            # --   --
            prev_channels = channels
            skip = (ksz - stride) * dilation_factor
            pad_left += total_scale * skip
            logger.log(f'pad += {total_scale} * {ksz-stride} * {dilation_factor}')
            # Register
            self.skips.append(skip)
            total_scale *= stride

        self.pad_left = pad_left
        self.total_scale = total_scale

        self.final_conv_0 = nn.Conv1d(channels, channels, 1)
        self.final_conv_0.bias.data.zero_()
        self.final_conv_1 = nn.Conv1d(channels, channels, 1)
        # We don't set the bias to 0 here because otherwise the initial model
        # would produce the 0 vector when the input is 0, and it will make
        # the vq layer unhappy.

    def forward(self, samples):
        """
        Args:
            samples: (N, samples_i) numeric tensor
        Returns:
            (N, samples_o, channels) numeric tensor
        """
        x = samples.unsqueeze(1)
        #logger.log(f'sd[samples] {x.std()}')

        for i, stuff in enumerate(zip(self.convs_wide, self.convs_1x1, self.layer_specs, self.skips)):
            # parts
            conv_wide, conv_1x1, layer_spec, skip = stuff
            stride, ksz, dilation_factor = layer_spec

            # x: loop input/output
            # x -> conv_wide -> tanh*sigmoid -> 1x1 Conv -> Res -> x
            # |______________________________________________|

            x1 = conv_wide(x)
            #logger.log(f'sd[conv.s] {x1.std()}')
            x1_a, x1_b = x1.split(x1.size(1) // 2, dim=1)
            # Gated Convolutional Layers (c.f. PixelRNN)
            x2 = torch.tanh(x1_a) * torch.sigmoid(x1_b)
            #logger.log(f'sd[act] {x2.std()}')
            x3 = conv_1x1(x2)
            #logger.log(f'sd[conv.1] {x3.std()}')
            if i == 0:
                x = x3
            else:
                x = x3 + x[:, :, skip:skip+x3.size(2)*stride].view(x.size(0), x3.size(1), x3.size(2), -1)[:, :, :, -1]
            #logger.log(f'sd[out] {x.std()}')

        # 1x1 Conv -> ReLU -> 1x1 Conv
        x = self.final_conv_1(F.relu(self.final_conv_0(x)))

        #logger.log(f'sd[final] {x.std()}')
        # (N, C, T) -> (N, T, C)
        return x.transpose(1, 2)
