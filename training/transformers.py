# code inspired by https://github.com/wpeebles/gangealing/blob/main/models/spatial_transformers/spatial_transformer.py
# and https://github.com/wpeebles/gangealing/blob/main/models/spatial_transformers/warping_heads.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import math 
from training.networks_nostyle import ConvLayer, ResBlock
from training.networks_stylegan2 import FullyConnectedLayer
from training.antialiased_sampling import MipmapWarp, Warp


default_camera_dist = 2.1
K = torch.tensor([[2.1326, 0, 0, 0],
                 [0, 2.1326, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
default_cam = torch.tensor([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, -default_camera_dist]])

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
    def __init__(self):
        super().__init__()

        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
    
    def encode_features(self, input_img):
        pass

    def render(self, planes, mat, resolution):
        N = planes.shape[0]
        H = W = resolution
        
        planes = planes.view(N, 3, -1, planes.shape[-2], planes.shape[-1])

        device = planes.device

        depth = None

        # ------- ray sampling --------
        # TODO
        K_mat = K.to(device)
        invK_mat = torch.linalg.inv(K_mat)
        invK_2d_grid = F.affine_grid(invK_mat[:3,:3].unsqueeze(0).repeat(N,1,1), (N, 2, H, W))     # (N, H, W, 2)
        grid = torch.cat([invK_2d_grid, torch.ones_like(invK_2d_grid)], dim=-1)             # (N, H, W, 4)

        inv_mat = torch.linalg.inv(convert_square_mat(mat))
        ray_dirs = (inv_mat[:, None, None] @ grid.unsqueeze(-1)).squeeze(-1)
        ray_origins = (inv_mat @ invK_mat)[:, None, None, :3, -1].repeat(1, H, W, 1)

        # ------- rendering -----------

        _, depth, _ = self.renderer(planes, self.decoder, ray_origins, ray_dirs, self.rendering_options)

        return depth


class Transformer(nn.Module):
    def __init__(self, input_size, channel_multiplier=0.5, num_heads=1, antialias=True):
        super().__init__()

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, int(self.channels[input_size]), input_size, kernel_size=1)]

        log_size = int(math.log(input_size, 2))
        log_downsample = int(math.log(8, 2))

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

            convs.append(ResBlock(int(in_channel), int(out_channel), img_size, 3, up=up))

            in_channel = out_channel

        # final_conv
        convs = convs + [ConvLayer(in_channel, self.channels[4], img_size, 3)]

        self.convs = nn.Sequential(*convs)
        self.num_heads = num_heads
        self.warper = MipmapWarp(max_num_levels = 3.5) if antialias else Warp()


    def encode_features(self, input_img):
        return self.convs(input_img)

    def forward(self, input_img, source_img=None, padding_mode='border', alpha=None, iters=1, return_full=False, prev_mat=None, **kwargs):
        if source_img == None:
            source_img = input_img

        out = input_img
        prev_mat = None
        depth = None

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
    def __init__(self, input_size, channel_multiplier=0.5, num_heads=1, antialias=True, intialize_zero=True):
        super().__init__(input_size, channel_multiplier, num_heads, antialias)

        self.final_linear = FullyConnectedLayer(self.channels[4] * 4 * 4, self.channels[4], activation='lrelu')

        self.really_final_linear = FullyConnectedLayer(self.channels[4], 2 * self.num_heads)

        if False: #initialize_zero:
            self.really_final_linear.weight.data.zero_()
            self.really_final_linear.bias.data.zero_()

        self.triplane_block = TriplaneBlock()
    
    def encode_features(self, input_img):
        out = self.convs(input_img)
        out = self.final_linear(out.view(out.shape[0], -1))

        return out

    def forward(self, input_img, source_img=None, padding_mode='border', alpha=None, iters=1, return_full=False, prev_mat=None, use_initial_depth=False, **kwargs):
        if source_img == None:
            source_img = input_img

        out = input_img
        prev_mat = None
        depth = None

        for i in range(iters):
            out, prev_mat, _depth = self.single_forward(out, source_img, padding_mode, alpha, return_full=True, prev_mat=prev_mat, depth=depth, **kwargs)
            
            if (use_initial_depth and i == 1) or not use_initial_depth:
                depth = _depth

        if return_full:
            return out, prev_mat, depth
        else:
            return out

    def single_forward(self, input_img, source_img=None, padding_mode='border', alpha=None, return_full=False, prev_mat=None, depth=None, **kwargs):
        if source_img == None: 
            source_img = input_img

        N, C, H, W = source_img.shape 
        device = input_img.device

        features = self.encode_features(input_img)
        params = self.really_final_linear(features)

        # TODO
        mat = create_affine_mat3D(*([torch.zeros_like(params[:, :1])]*4), *torch.split(params, 1, dim=1)) 

        if prev_mat != None:
            if prev_mat.shape[-2] == 2:
                # TODO
                prev_mat = upgrade_2x3_to_3x4(prev_mat)                     
        else:
            prev_mat = torch.linalg.inv(convert_square_mat(default_cam.unsqueeze(0).repeat(N,1,1))).to(device)[..., :3, :4]

        mat = prev_mat @ convert_square_mat(mat)        # (N, 3, 4)
        
        # TODO
        K_mat = K.to(device)                                                    
        invK_mat = torch.linalg.inv(K_mat)
        invK_2d_grid = F.affine_grid(invK_mat[:2,:3].unsqueeze(0).repeat(N,1,1), (N, C, H, W), align_corners=False)#.to(device)     # (N, H, W, 2)

        if depth == None:
            # TODO: utilize triplaneblock
            #depth = torch.ones_like(input_img.permute(0,2,3,1)[..., :1]) * default_camera_dist
            depth = default_camera_dist + invK_2d_grid.pow(2).sum(dim=-1,keepdim=True)

        grid = torch.cat([invK_2d_grid * depth, 
                        depth, 
                        torch.ones_like(invK_2d_grid[..., :1])], dim=-1)  # (N, H, W, 4)

        default_cam_ = convert_square_mat(default_cam[None]).to(device)

        grid = ((K_mat[None, :3, :3] @ mat @ default_cam_)[:, None, None] @ grid.unsqueeze(-1)).squeeze(-1)

        warped_depth = grid[..., 2:3]
        grid = grid[..., :2] / (warped_depth + 1e-6)

        out = self.warper(source_img, grid, padding_mode=padding_mode)

        if return_full:
            return out, mat, depth
        
        else:
            return out



class SimilarityTransformer(Transformer):
    def __init__(self, input_size, channel_multiplier=0.5, num_heads=1, antialias=True, intialize_zero=True):
        super().__init__(input_size, channel_multiplier, num_heads, antialias)

        self.final_linear = FullyConnectedLayer(self.channels[4] * 4 * 4, self.channels[4], activation='lrelu')
        self.really_final_linear = FullyConnectedLayer(self.channels[4], 4 * self.num_heads)

        if False: #initialize_zero:
            self.really_final_linear.weight.data.zero_()
            self.really_final_linear.bias.data.zero_()
    
    def encode_features(self, input_img):
        out = self.convs(input_img)
        out = self.final_linear(out.view(out.shape[0], -1))

        return out

    def single_forward(self, input_img, source_img=None, padding_mode='border', alpha=None, return_full=False, prev_mat=None, **kwargs):
        if source_img == None: 
            source_img = input_img

        N, C, H, W = source_img.shape 
        device = input_img.device

        features = self.encode_features(input_img)
        params = self.really_final_linear(features)
        
        depth = torch.ones_like(input_img[..., :1]) * default_camera_dist
        mat = create_affine_mat2D(*torch.split(params, 1, dim=1)) # TODO
        print(mat)
        if prev_mat != None:
            mat = prev_mat @ convert_2x3_3x3(mat)

        grid = F.affine_grid(mat, (N, C, H, W), align_corners=False).to(device)
        out = self.warper(source_img, grid, padding_mode=padding_mode)

        if return_full:
            return out, mat, depth
        
        else:
            return out