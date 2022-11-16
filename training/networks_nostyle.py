import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch_utils import persistence
from torch_utils import misc
from training.networks_stylegan2 import modulated_conv2d, Conv2dLayer
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma


# from SynthesisLayer
@persistence.persistent_class
class ConvLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        #w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.kernel_size = kernel_size

        #self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = int(self.resolution / self.up)
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])
        styles = torch.ones((x.shape[0], x.shape[1]), device=x.device) / np.sqrt(self.in_channels * self.kernel_size * self.kernel_size) #self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        kwargs = {'up': self.up} if self.up >= 1 else {'down': int(1 / self.up)}
        
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, #up=self.up, 
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv,
            demodulate=False,
            **kwargs)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}', 
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])


# modified from SynthesisBlock, with architecture fixed as 'resnet'
@persistence.persistent_class
class ResBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        #w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        up = 2,
        #is_last,                                # Is this the last block?
        #architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        #assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        #self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.up = up
        #self.is_last = is_last
        #self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = ConvLayer(in_channels, out_channels, resolution=resolution, up=up,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = ConvLayer(out_channels, out_channels, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if in_channels != 0: #and architecture == 'resnet':
            kwargs = {'up': self.up} if self.up >= 1 else {'down': int(1 / self.up)}
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False,
                resample_filter=resample_filter, channels_last=self.channels_last,
                **kwargs)

    def forward(self, x, force_fp32=False, fused_modconv=None, update_emas=False, **layer_kwargs):
        if x.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            input_res = int(self.resolution / self.up)
            misc.assert_shape(x, [None, self.in_channels, input_res, input_res])
            x = x.to(dtype=dtype, memory_format=memory_format)

        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x, fused_modconv=fused_modconv, **layer_kwargs)
        x = self.conv1(x, fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
        x = y.add_(x)

        assert x.dtype == dtype
        return x#, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'