# code inspired by https://github.com/wpeebles/gangealing/blob/main/models/spatial_transformers/spatial_transformer.py
# and https://github.com/wpeebles/gangealing/blob/main/models/spatial_transformers/warping_heads.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from torchvision.transforms.functional import gaussian_blur, center_crop
import math 
import numpy as np

from typing import List, Optional, Dict

#from training.networks_nostyle import ConvLayer, ResBlock
from training.networks_stylegan2 import FullyConnectedLayer
from training.antialiased_sampling import MipmapWarp, Warp
from training.volumetric_rendering.renderer import ImportanceRenderer
from torch_utils.ops import upfirdn2d

from training.rosinality.networks import ConvLayer, ResBlock, EqualLinear

# TODO: move the constants somewhere else
default_camera_dist = 2.1

K = torch.tensor([[2.1326, 0, 0, 0],
                 [0, 2.1326, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

default_cam = torch.tensor([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, -default_camera_dist]])

# from warping_heads.SimilarityHead
def create_affine_mat2D(rot, scale, shift_x, shift_y):
    # This function takes the raw output of the parameter regression network and converts them into
    # an affine matrix representing the predicted similarity transformation.
    # Inputs each of size (N, K), K = number of heads/clusters
    N, K = rot.size()
    rot = torch.tanh(rot) * math.pi
    scale = torch.exp(scale)
    cos_rot = torch.cos(rot)
    sin_rot = torch.sin(rot)
    
    matrix = [scale * cos_rot, -scale * sin_rot, shift_x,
                scale * sin_rot, scale * cos_rot, shift_y]
    matrix = torch.stack(matrix, dim=2)  # (N, K, 6)
    matrix = matrix.reshape(N, 2, 3)  # (N, 2, 3)

    return matrix

def create_affine_mat3D(roll, dist, u, v, elev, azim):
    # This function takes the raw output of the parameter regression network and converts them into
    # an affine matrix representing the predicted similarity transformation.
    # Inputs each of size (N, K), K = number of heads/clusters
    # code referenced from pytorch3d : camera_position_from_spherical_angles, and look_at_rotation
    # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/cameras.html#CamerasBase
    # creates transform matrix from aligned screen space to unaligned screen space
    N, _ = elev.size()
    device = elev.device
    one = torch.ones_like(elev)
    
    azim = torch.tanh(azim) * math.pi
    elev = torch.tanh(elev) * math.pi      
    
    up = torch.tensor([0, 1, 0], dtype=torch.float).repeat([N, 1]).to(device)
    
    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    
    camera_position = torch.cat([x, y, z], dim=1)
    
    if camera_position.dim() == 0:
        camera_position = camera_position.view(1, -1) # add batch dims
            
    z_axis = F.normalize(camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    T = torch.stack([-x, -y, -z+one], dim=-1)
    T = T * 0.0
                    
    RT = torch.cat([R, T], dim=1).transpose(1, 2)
    matrix = RT.reshape(N, 3, 4) # (N, 3, 4)

    return matrix

def convert_square_mat(matrix):
    if matrix.shape[-1] == matrix.shape[-2]:
        return matrix

    zeros = torch.zeros_like(matrix[..., :1, :-1])
    one = torch.ones_like(matrix[..., :1, :1])
    zeros_and_one = torch.cat([zeros, one], dim=-1)

    square_mat = torch.cat([matrix, zeros_and_one], dim=-2)

    return square_mat

def upgrade_2Dmat_to_3Dmat(matrix):
    R = matrix[..., :2, :2]         # from roll
    T = matrix[..., :2, -1:]        # from trans_x, and trans_y
    
    scale = torch.linalg.norm(R[..., 0, :].view(-1, 2), dim=-1).view(*matrix.shape[:-2])
    scale = 1 / scale * default_camera_dist 
    
    zero = torch.zeros_like(scale)
    one = torch.ones_like(scale)
    
    T = scale[..., None, None]*T / default_camera_dist
    R = F.normalize(R, dim=-1)

    M_2x4 = torch.cat([R,  torch.zeros_like(T), T], dim=-1)   
    M_1x4 = torch.stack([zero, zero, one, scale], dim=-1).view(*matrix.shape[:-2], 1, 4)        
    M_3x4 = torch.cat([M_2x4, M_1x4], dim=-2)
    
    return M_3x4

def create_mat3D_from_gen_z(gen_z):
    params_label = gen_z[:, :6]
    mat2d_label = create_affine_mat2D(*torch.split(params_label[:, :4], 1, dim=1))
    mat3d_label = create_affine_mat3D(None,None,None,None, *torch.split(params_label[:, 4:], 1, dim=1))
    mat_label = upgrade_2Dmat_to_3Dmat(mat2d_label) @ convert_square_mat(mat3d_label)

    return mat_label

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


class TriplaneBlock(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.input_size = input_size
        self.plane_features = 32
        self.color_features = 32
        self.resizer = Resize(input_size)
        """
        # nvidia
        self.encoder = nn.Sequential(
                            ConvLayer(3, 128, input_size, kernel_size=1),
                            #ResBlock(128, 128, input_size, 3, up=1),
                            ResBlock(128, 256, input_size //2, 3, up=0.5),
                            ResBlock(256, 256, input_size //2, 3, up=1),
                            ConvLayer(256, 3 * self.plane_features, input_size //2, kernel_size=1),
                        )
        """

        """
        IDEA: formulate encoder as...
        (fixed GAN layers; ~5 layers) + (the rest of GAN layer)
                                           ^ stylization by latent code from img encdoer

        this way, we can inject a shape prior into the canonical volume space
        """

        # rosinality
        self.encoder = nn.Sequential(
                            ConvLayer(3, 128, 1),
                            ResBlock(128, 128, downsample=False),
                            ResBlock(128, 256, downsample=True),
                            ResBlock(256, 256, downsample=False),
                            ConvLayer(256, 3 * self.plane_features, 1),
                        )

        self.decoder = OSGDecoder(self.plane_features, {'decoder_lr_mul': 0.001, 'decoder_output_dim': self.color_features})

        self.renderer = ImportanceRenderer()

        # TODO: feed options to the block via init argument, move constants somewhere else
        self.rendering_options = {
            'depth_resolution': 24,
            'depth_resolution_importance': 24,
            'ray_start': 2.25 - 0.6,
            'ray_end': 3.3 - 0.6,
            'disparity_space_sampling': False,
            'box_warp': 1,
            'clamp_mode': 'softplus',
            'white_back': False,
            #'resolution': 32,
        }
    
    def encode_features(self, input_img):
        input_img = self.resizer(input_img)
        features = self.encoder(input_img)

        return features

    def render(self, planes, resolution, mat):
        # mat: world -> cam

        N = planes.shape[0]
        H = W = resolution

        planes = planes.view(planes.shape[0], 3, -1, planes.shape[-2], planes.shape[-1])

        device = planes.device

        depth = None

        # ------- ray sampling --------

        # invK_2d_grid : screen -> cam
        K_mat = K.to(device)
        invK_mat = torch.linalg.inv(K_mat)
        invK_2d_grid = F.affine_grid(invK_mat[:2,:3].unsqueeze(0), (1, 2, H, W))     # (1, H, W, 2)

        """
        we multiply 0.5 due to L61 on volumetric_rendering: coordinates = (2/box_warp) * coordinates
        original EG3D code starts from (-0.5, 0.5) range for [H,W] box, 
        while we use (-1, 1) which is standard for F.affine_grid + F.grid_sample

        range history for EG3D: 0.5 (initial) -> 1 (due to L61)
        range history for our code: 1 (initial) -> 0.5 (by following code) -> 1 (due to L61)
        """

        grid = torch.cat([0.5*invK_2d_grid, torch.ones_like(invK_2d_grid)], dim=-1).expand(N, -1, -1, -1)                # (N, H, W, 4)

        # inv_mat: cam -> world
        inv_mat = torch.linalg.inv(convert_square_mat(mat))[..., :3, :4].expand(N, -1, -1)

        ray_dirs = (inv_mat[:, None, None] @ grid.unsqueeze(-1)).squeeze(-1).view(N, -1, 3)
        ray_origins = (inv_mat @ invK_mat.unsqueeze(0))[:, None, None, :3, -1].expand(-1, H, W, -1).view(N, -1, 3)

        # ------- rendering -----------

        colors, depth, weights = self.renderer(planes, self.decoder, ray_origins, ray_dirs, self.rendering_options)

        colors = colors.view(N, H, W, self.color_features)
        depth = depth.view(N, H, W, 1)

        return colors, depth, weights

    def forward(self, input_img, resolution=None, mat=None):
        device = input_img.device

        if resolution == None:
            resolution = input_img.shape[2]

        if True: #mat == None:
            N = input_img.shape[0]
            inv_mat = default_cam.unsqueeze(0).repeat(N,1,1).to(device)
            mat = torch.linalg.inv(convert_square_mat(inv_mat))

        planes = self.encode_features(input_img)
        colors, depth, weights = self.render(planes, resolution, mat)

        return colors, depth, weights


class Transformer(nn.Module):
    def __init__(self, input_size, channel_multiplier=0.5, num_heads=1, antialias=True, fixed_params=None):
        super().__init__()

        self.channels = {
            4: 256,
            8: 256,
            16: 256,
            32: 128,
            64: 128 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        blur_kernel=[1, 3, 3, 1]

        #convs = [ConvLayer(3, int(self.channels[input_size]), input_size, kernel_size=1)]  # nvidia
        convs = [ConvLayer(3, int(self.channels[input_size]), 1)] # rosinality

        log_size = int(math.log(input_size, 2))

        in_channel = self.channels[input_size]

        end_log = 2 #log_size - 4 if self.is_flow else 2
        assert end_log >= 0

        num_downsamples = 0
        for i in range(log_size, end_log, -1):
            downsample = True #(not self.is_flow) or (num_downsamples < log_downsample)
            num_downsamples += downsample
            img_size = 2 ** (i-1)
            up = 1 if not downsample else 0.5
            out_channel = self.channels[img_size]

            #convs.append(ResBlock(int(in_channel), int(out_channel), img_size, 3, up=up))      # nvidia
            convs.append(ResBlock(int(in_channel), int(out_channel), blur_kernel, downsample)) # rosinality

            in_channel = out_channel

        # final_conv
        #convs = convs + [ConvLayer(in_channel, self.channels[4], img_size, 3)]
        convs = convs + [ConvLayer(in_channel, self.channels[4], 3)]     # rosinality

        self.convs = nn.Sequential(*convs)
        self.num_heads = num_heads
        self.warper = MipmapWarp(max_num_levels = 3.5) if antialias else Warp()

        self.fixed_params = fixed_params

    def encode_features(self, input_img):
        return self.convs(input_img)

    def infer_params(self, input_img):
        if self.fixed_params != None:
            return self.fixed_params
        else:
            raise NotImplementedError()

    def forward(self, input_img, source_img=None, padding_mode='border', alpha=None, iters=1, return_full=False, prev_mat=None, depth=None, **kwargs):
        if source_img == None:
            source_img = input_img

        out = input_img

        for i in range(iters):
            out, prev_mat, depth = self.single_forward(out, source_img, padding_mode, alpha, return_full=True, prev_mat=prev_mat, depth=depth, **kwargs)

        if return_full:
            return out, prev_mat, depth
        else:
            return out

    def single_forward(self, input_img, source_img=None, padding_mode='border', alpha=None, return_full=False, prev_mat=None, **kwargs):
        # should be implemented by child classes

        raise NotImplementedError()

    def congeal_points(self, input_img, points, padding_mode='border', alpha=None, **kwargs):
        # TODO

        raise NotImplementedError()

    def uncongeal_points(self, input_img, points, padding_mode='border', alpha=None, **kwargs):
        # TODO

        raise NotImplementedError()



class PerspectiveTransformer(Transformer):
    def __init__(self, input_size, channel_multiplier=0.5, num_heads=1, antialias=True, fixed_params=None, initialize_zero=True):
        super().__init__(input_size, channel_multiplier, num_heads, antialias, fixed_params)

        #self.final_linear = FullyConnectedLayer(self.channels[4] * 4 * 4, self.channels[4], activation='lrelu')    # nvidia
        self.final_linear = EqualLinear(self.channels[4] * 4 * 4, self.channels[4], activation='fused_lrelu')     # rosinality
        self.really_final_linear = nn.Linear(self.channels[4], 2 * self.num_heads)

        if initialize_zero:
            self.really_final_linear.weight.data.zero_()
            self.really_final_linear.bias.data.zero_()

        #if True: #initialize_zero:
        #    self.really_final_linear.weight.data.mul_(0.5)
        #    #self.really_final_linear.bias.data.mul_(0.5)
        #    self.really_final_linear.bias.data.zero_()

        # encoder - decoder 
        self.triplane_block = TriplaneBlock(256)
    
    def encode_features(self, input_img):
        out = self.convs(input_img)
        out = self.final_linear(out.view(out.shape[0], -1))

        return out
    
    def infer_params(self, input_img):
        if self.fixed_params != None:
            assert self.fixed_params.shape[-1] == self.really_final_linear.weight.data.shape[0]

            return self.fixed_params
        else:
            features = self.encode_features(input_img)
            params = self.really_final_linear(features)

            #print(params.std(0), params.mean(0), self.really_final_linear.bias.data)

            return params

    def forward(self, input_img, source_img=None, padding_mode='border', alpha=None, iters=1, return_full=False, prev_mat=None, depth=None, use_initial_depth=False, **kwargs):
        if source_img == None:
            source_img = input_img

        out = input_img

        for i in range(iters):
            out, prev_mat, _depth = self.single_forward(out, source_img, padding_mode, alpha, return_full=True, prev_mat=prev_mat, depth=depth, **kwargs)
            
            # TODO: batchfiy use_initial_depth
            if not use_initial_depth or (i == 0 and use_initial_depth):
                """
                [EG3D]'s triplane generator sometimes receives a GT pose prior, sometimes not
                if the prior is 100% reliable, the generator cheats by aligning the plane shape with the camera direction 
                (see Fig4 on EG3D supplemental mat)

                we have two choices for inferring the aligned volume 
                    1. inferring from the initial 2D-aligned img
                    2. iteratively refining the depth

                each method has a potential downfall:
                1: since the triplane is created from an image-to-image encoder, rather than a latent vector,
                the inferred depth has a tendency to resemble the input image

                2: if the initial depth is degenerate, the resulting image will also be degenerate
                since each iteration is performed via a single network, the network receives two groups of input:
                2D-cropped inputs from the first iteration, which are relatively "real"
                and 3D-transformed inputs from further iterations, which may or may not be degenerate
                the network will also be corrupted by degenerate inputs

                1. has the least corrupted input, while 2. requires less transforming work from the network (assuming our iteration hasn't diverged)
                by mixing two approaches in the same way EG3D makes its pose prior half-reliable on purpose,
                we seek to estimate a better initial depth, while also utilizing the stabilizing property of iteration-based methods

                on unrelated note, [IDE-3D] also uses a hybrid GAN inversion: initialization + iteration.
                """

                # update depth if the conditions are met
                depth = _depth

        if return_full:
            return out, prev_mat, depth
        else:
            return out

    def single_forward(self, input_img, source_img=None, padding_mode='border', alpha=None, return_full=False, prev_mat=None, depth=None, blur_sigma=0,  **kwargs):
        if source_img == None: 
            source_img = input_img

        device = input_img.device

        # -------------- prepare camera matrix ---------------------------------------------
        params = self.infer_params(input_img)

        mat = create_affine_mat3D(*([torch.zeros_like(params[:, :1])]*4), *torch.split(params, 1, dim=1)) 
        mat = self.join_prev_mat(mat, prev_mat)

        # canonical camera for rendering the canonical depth
        default_cam_ = convert_square_mat(default_cam[None]).to(device)
        render_mat = torch.linalg.inv(default_cam_)[..., :3, :4]

        out, depth = self.render_and_warp(input_img, mat, render_mat, source_img=source_img, depth=depth, padding_mode=padding_mode, blur_sigma=blur_sigma)

        if return_full:
            return out, mat, depth
        
        else:
            return out

    def siamese_forward(self, out_t, out_s, source_img, prev_mat_t=None, prev_mat_s=None, padding_mode='border', **kwargs):
        params_s = self.infer_params(out_s)

        mat_s = create_affine_mat3D(*([torch.zeros_like(params_s[:, :1])]*4), *torch.split(params_s, 1, dim=1))
        mat_s = self.join_prev_mat(mat_s, prev_mat_s)


        params_t = self.infer_params(out_t)

        mat_t = create_affine_mat3D(*([torch.zeros_like(params_t[:, :1])]*4), *torch.split(params_t, 1, dim=1))
        mat_t = self.join_prev_mat(mat_t, prev_mat_t)


        out, _ = self.render_and_warp(out_t, mat_s, mat_t,  source_img=source_img, **kwargs)

        return out


    def join_prev_mat(self, mat, prev_mat=None):
        if prev_mat != None:
            if prev_mat.shape[-2] == 2:
                prev_mat = upgrade_2Dmat_to_3Dmat(prev_mat)                     
        else:
            # default_cam: cam1 -> world
            default_cam_ = default_cam.unsqueeze(0).repeat(N,1,1).to(mat.device)
            default_cam_ = convert_square_mat(default_cam_)

            # inv_default_cam: world -> cam1
            prev_mat = torch.linalg.inv(default_cam_)[..., :3, :4]

        mat = prev_mat @ convert_square_mat(mat)        # (N, 3, 4)

        return mat

    def render_and_warp(self, input_img, mat, render_mat, source_img=None, depth=None, padding_mode='border', blur_sigma=0):
        if source_img == None: 
            source_img = input_img

        N, C, H, W = source_img.shape 
        device = input_img.device

        # --------------- prepare canonical depth ------------------------------------------

        """
        We assume truncated img with psi to be the canon; 
        since psi changes as the training proceeds, the output transform is relative to the current psi-img, rather than being absolute
        Rather than relying on such assumption, we could adapt siamese architecture, and use the inferred param of the target img to render the depth
        """

        # if we're not given an input depth, render the depth from scratch
        if depth == None:
            _, depth, _ = self.triplane_block(input_img, 64, mat=render_mat)

            """
            in an ablation study of [StyleNeRF], without progressive learning, concave depths are created
            [EG3D] Progressive Training blurs the image fed into the discriminator in early epochs, 
            to reproduce the effect of progressive learning without having to modify the number of layers in midst of training

            in early epochs, depth has noises which forces the output to be in lower res, and confuse the perceptual loss
            by filtering out the noises with a blur kernel (e.g. gaussian), we can focus on the coarse geometry
            """

            # sigma from https://pytorch.org/vision/main/generated/torchvision.transforms.functional.gaussian_blur.html
            # TODO: design sigma progression graph
            depth = depth.permute(0,3,1,2)

            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                with torch.autograd.profiler.record_function('blur'):
                    """
                    without reflection padding, spatial inductive bias is introduced to depth image around the border 
                    (see Positional Encoding as Spatial Inductive Bias in GANs for the effect of zero-padding)
                    """
                    original_size = depth.shape[2:]
                    blur_size_int = int(blur_size)
                    depth = F.pad(depth, (blur_size_int,blur_size_int,blur_size_int,blur_size_int), "reflect")

                    f = torch.arange(-blur_size, blur_size + 1, device=depth.device).div(blur_sigma).square().neg().exp2()
                    depth = upfirdn2d.filter2d(depth, f / f.sum())

                    depth = center_crop(depth, original_size)

            """
            RELATED WORK:
                due to memory constraints, volume rendering is limited for low-res
                prev works [StyleNeRF, StyleSDF, EG3D, ...] make up for low-res by post-aggregation 2D upsampling
                the critical downside is that 3D priors are lost after aggregation, thus losing multi-view consistency
                this is compenstated by NeRF path regularization [StyleNeRF], dual-discrimination [EG3D], or double discriminators [StyleSDF]
                all of which serve to enforce consistency loss between the actual NeRF render and the upsampled render
            
            IDEA: 
                instead of comparing full renders, maybe focus on high-frequency patches??
                1 pixel in low-res = 2/4/8 pixel patch in high res
                selectively render a potentionally high-res patch, 
                do a switchy-swap with convex-upsampled (RAFT?) patch, or enforce consistency...
            """

            depth = F.interpolate(depth, (H, W), mode='bilinear')
            depth = depth.permute(0,2,3,1)


        # --------------- prepare sampling grid ---------------------------------------------

        """
        apply K inverse: screen -> cam1, by affine_grid
        """

        K_mat = K.to(device)                                                    
        invK_mat = torch.linalg.inv(K_mat)
        invK_2d_grid = F.affine_grid(invK_mat[:2,:3].unsqueeze(0).repeat(N,1,1), (N, C, H, W), align_corners=False)     # (N, H, W, 2)

        ones = torch.ones_like(invK_2d_grid[..., :1])
        grid = torch.cat([invK_2d_grid * depth, 
                        depth, 
                        ones], dim=-1)  # (N, H, W, 4)

        # ---------------- transform the grid: cam1 -> world -> cam2 -> screen --------------
        """
        inv(render_mat) : cam1 -> world
        mat: world -> cam2
        K_mat : cam2 -> screen
        """

        #print(mat.std(0))
        inv_render_mat = torch.linalg.inv(convert_square_mat(render_mat))
        grid = ((K_mat[None, :3, :3] @ mat @ inv_render_mat)[:, None, None] @ grid.unsqueeze(-1)).squeeze(-1)

        warped_depth = grid[..., 2:3]
        grid = grid[..., :2] / (warped_depth + 1e-6)

        out = self.warper(source_img, grid, padding_mode=padding_mode)

        return out, depth

    



class SimilarityTransformer(Transformer):
    def __init__(self, input_size, channel_multiplier=0.5, num_heads=1, antialias=True, fixed_params=None, initialize_zero=True):
        super().__init__(input_size, channel_multiplier, num_heads, antialias, fixed_params)

        #self.final_linear = FullyConnectedLayer(self.channels[4] * 4 * 4, self.channels[4], activation='lrelu')    # nvidia
        self.final_linear = EqualLinear(self.channels[4] * 4 * 4, self.channels[4], activation='fused_lrelu')     # rosinality
        self.really_final_linear = nn.Linear(self.channels[4], 4 * self.num_heads)

        if initialize_zero:
            self.really_final_linear.weight.data.zero_()
            self.really_final_linear.bias.data.zero_()

        #if True: #initialize_zero:
        #    self.really_final_linear.weight.data.mul_(0.5)
        #    #self.really_final_linear.bias.data.mul_(0.5)
        #    self.really_final_linear.bias.data.zero_()
    
    def encode_features(self, input_img):
        out = self.convs(input_img)
        out = self.final_linear(out.view(out.shape[0], -1))

        return out

    def infer_params(self, input_img):
        if self.fixed_params != None:
            assert self.fixed_params.shape[-1] == self.really_final_linear.weight.data.shape[0]

            return self.fixed_params
        else:
            features = self.encode_features(input_img)
            params = self.really_final_linear(features)

            return params

    def join_prev_mat(self, mat, prev_mat=None):
        if prev_mat != None:
            mat = prev_mat @ convert_square_mat(mat)
        
        return mat

    def single_forward(self, input_img, source_img=None, padding_mode='border', alpha=None, return_full=False, prev_mat=None, **kwargs):
        if source_img == None: 
            source_img = input_img

        N, C, H, W = source_img.shape 
        device = input_img.device

        params = self.infer_params(input_img)
        
        mat = create_affine_mat2D(*torch.split(params, 1, dim=1))
        mat = self.join_prev_mat(mat, prev_mat)
        
        grid = F.affine_grid(mat, (N, C, H, W), align_corners=False).to(device)
        out = self.warper(source_img, grid, padding_mode=padding_mode)

        if return_full:
            return out, mat, None
        
        else:
            return out
    

    def siamese_forward(self, out_t, out_s, source_img, prev_mat_t=None, prev_mat_s=None, padding_mode='border', **kwargs):
        N, C, H, W = source_img.shape
        device = out_t.device

        params_s = self.infer_params(out_s)
        
        mat_s = create_affine_mat2D(*torch.split(params_s, 1, dim=1))
        mat_s = self.join_prev_mat(mat_s, prev_mat_s)

        params_t = self.infer_params(out_t)

        mat_t = create_affine_mat2D(*torch.split(params_t, 1, dim=1))
        mat_t = self.join_prev_mat(mat_t, prev_mat_t)

        inv_mat_t = torch.linalg.inv(convert_square_mat(mat_t))

        grid = F.affine_grid(mat_s @ inv_mat_t, (N, C, H, W), align_corners=False).to(device)
        out = self.warper(source_img, grid, padding_mode=padding_mode)

        return out


class TransformerSequence(nn.Module):
    def __init__(self, transformers: List[Transformer]):
        super().__init__()
        self.transformers = nn.ModuleList(transformers)
    
    def forward(self, input_img, source_img=None, padding_mode='border', alpha=None, return_full=False, prev_mat=None, depth=None, use_initial_depth=False, **kwargs):
        if source_img == None:
            source_img = input_img

        out = input_img
        kwargs['return_full'] = True

        for transformer in self.transformers:
            out, prev_mat, depth = transformer(out, source_img=source_img, prev_mat=prev_mat, depth=depth, **kwargs)

        if return_full:
            return out, prev_mat, depth
        else:
            return out

    
    def siamese_forward(self, target_img, source_img, padding_mode='border', alpha=None, return_full=False, 
                        use_initial_depth=False, **kwargs):
        
        kwargs['return_full'] = True
        prev_mat_s = depth_s = None
        prev_mat_s = depth_t = None

        out_s = source_img
        out_t = target_img
        """
        align to the canonical until the last transformer module
        in the last module, render with target img's matrix, and warp to the source img
        """

        for transformer in self.transformers[:-1]:
            out_s, prev_mat_s, depth_s = transformer(out_s, source_img=source_img, prev_mat=prev_mat_s, depth=depth_s, **kwargs)
            out_t, prev_mat_t, depth_t = transformer(out_t, source_img=target_img, prev_mat=prev_mat_t, depth=depth_t, **kwargs)

        last_transformer = self.transformers[-1]
        
        #last_transformer.siamese_forward(out_s, out_t, target_img, prev_mat_s, prev_mat_t)

        return last_transformer.siamese_forward(out_t, out_s, source_img, prev_mat_t, prev_mat_s)


    def __getitem__(self, i):
        return self.transformers[i]
