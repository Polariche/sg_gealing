# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.transformers import create_mat3D_from_6params, convert_square_mat

import math


#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


#----------------------------------------------------------------------------

class TransformerSiameseLoss(Loss):
    def __init__(self, device, G, T, vgg, augment_pipe=None, use_initial_depth_prob=0, blur_init_sigma=0, blur_fade_kimg=0, psi_anneal=2000, epsilon=1e-4, pose_layers=5, fix_w_dist = False, pose_trunc_dist = 1, ):
        super().__init__()

        print("HI")

        self.device = device
        self.G = G          # Generator
        self.T = T          # Transformer
        self.vgg = vgg

        self.use_initial_depth_prob = use_initial_depth_prob 
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.psi_anneal         = psi_anneal
        self.augment_pipe       = augment_pipe
        self.epsilon            = epsilon
        self.pose_layers        = pose_layers
        self.pose_trunc_dist    = pose_trunc_dist
        self.fix_w_dist         = fix_w_dist


    def run_G(self, z, c, psi=None, update_emas=False):
        G = self.G
        w_avg = G.mapping.w_avg

        if self.fix_w_dist:
            ws = G.mapping(z, c, update_emas=update_emas)
            ws[:, :self.pose_layers] = w_avg + torch.nn.functional.normalize(ws[:, :self.pose_layers] - w_avg, dim=-1) * 1 * 10 * self.pose_trunc_dist
        else:
            ws = G.mapping(z, c, update_emas=update_emas)
            ws[:, :self.pose_layers] = w_avg.lerp(ws[:, :self.pose_layers], self.pose_trunc_dist)

        """
        ws_aligned = ws.clone()
        ws_align_dir = torch.nn.functional.normalize(torch.randn(ws_aligned[:, :self.pose_layers].shape, device=self.device)) * self.pose_trunc_dist * (1-psi)  * 10
        ws_align_dir = torch.sign((ws_align_dir * (ws[:, :self.pose_layers] - w_avg)).sum(dim=-1, keepdim=True)) * ws_align_dir 
        ws_aligned[:, :self.pose_layers] = ws[:, :self.pose_layers] + ws_align_dir
        

        ws_aligned = ws.clone()
        ws_aligned[:, :self.pose_layers] = w_avg.lerp(ws[:, :self.pose_layers], psi)
        """

        ws_posed = ws.clone()
        rand_ind = torch.randperm(ws.shape[0])
        ws_posed[:, :self.pose_layers] = ws[rand_ind, :self.pose_layers].lerp(ws[:, :self.pose_layers], psi)

        ws_aligned = ws.clone()
        ws_aligned[:, :self.pose_layers] = w_avg.lerp(ws[:, :self.pose_layers], 0)

        ws_input = torch.cat([ws, ws_posed, ws_aligned])

        img_set = G.synthesis(ws_input, update_emas=update_emas)

        img, img_posed, img_aligned = img_set.chunk(3)
        return img, img_posed, img_aligned,  ws, ws_posed, ws_aligned


    def run_T(self, img1, img2, blur_sigma=0, update_emas=False):
        #if self.augment_pipe is not None:
        #    img = self.augment_pipe(img)

        img1_new, img2_new, mat1, mat2, depth1, depth2 = self.T.siamese_forward(img1, img2, blur_sigma=blur_sigma, return_full=True, update_emas=update_emas)
        return img1_new, img2_new, mat1, mat2, depth1, depth2
        


    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        # set up constants
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        
        # cosine anneal, from gangealing.utils.annealing.cosine_anneal
        psi = 0.5 * (1 + torch.cos(torch.tensor(math.pi * min(cur_nimg//1000, self.psi_anneal) / self.psi_anneal)))
        psi = psi.to(self.device)

        training_stats.report('Loss/psi', psi)

        # run G & T
        with torch.autograd.profiler.record_function('Gmain_forward'):
            # run G to obtain a pair of images
            # img_1 is the "original", and img_2 is a pose-swapped version
            img_1,  img_2, img_aligned, ws_1, ws_2, ws_aligned = self.run_G(gen_z, gen_c, psi=psi)

            # siamese transformation
            transformed_1, transformed_2, mat_1, mat_2, _, _ = self.run_T(img_1.detach(), img_2.detach(), blur_sigma=blur_sigma)
            transformed_to_aligned, _ = self.T[1].render_and_warp(img_1.detach(), mat_1, None)
            img = torch.cat([img_1, img_2, img_aligned, 
                            transformed_1, transformed_2, transformed_to_aligned], dim=0)
            
            lpips_t0, lpips_t1 = self.vgg(img, resize_images=False, return_lpips=True).chunk(2)
            perceptual_loss = (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
            training_stats.report('Loss/perceptual_loss', perceptual_loss)


            # draw random pose from a distribution
            random_params = torch.randn((*img_1.shape[:-3], 6), device=self.device)
            param_weights = torch.tensor([0.1, 0.05, 0.4, 0.4, 0.05, 0.05], device=self.device)
            random_mat = create_mat3D_from_6params(random_params * param_weights)

            # render 
            new_img = self.T(img_1.detach(), render_mat=random_mat, blur_sigma=blur_sigma)
            _, random_mat_hat, _ = self.T(new_img, return_full=True, blur_sigma=blur_sigma)

            mat_loss = (convert_square_mat(random_mat_hat) @ torch.linalg.inv(convert_square_mat(random_mat)) - torch.eye(4).unsqueeze(0).to(self.device)).square().sum(-1).sum(-1)
            training_stats.report('Loss/mat_loss', mat_loss)

            loss = perceptual_loss.mean() + mat_loss.mean() * 1e-1


        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss.mul(gain).backward()

#----------------------------------------------------------------------------

class TransformerLoss(Loss):
    def __init__(self, device, G, T, vgg, augment_pipe=None, use_initial_depth_prob=0, blur_init_sigma=0, blur_fade_kimg=0, psi_anneal=2000, epsilon=1e-4, pose_layers=5, fix_w_dist = False, pose_trunc_dist = 1, ):
        super().__init__()

        print("HI")

        self.device = device
        self.G = G          # Generator
        self.T = T          # Transformer
        self.vgg = vgg

        self.use_initial_depth_prob = use_initial_depth_prob 
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.psi_anneal         = psi_anneal
        self.augment_pipe       = augment_pipe
        self.epsilon            = epsilon
        self.pose_layers        = pose_layers
        self.pose_trunc_dist    = pose_trunc_dist
        self.fix_w_dist         = fix_w_dist


    def run_G(self, z, c, psi=None, update_emas=False):
        G = self.G
        w_avg = G.mapping.w_avg

        if self.fix_w_dist:
            ws = G.mapping(z, c, update_emas=update_emas)
            ws[:, :self.pose_layers] = w_avg + torch.nn.functional.normalize(ws[:, :self.pose_layers] - w_avg, dim=-1) * 10 * self.pose_trunc_dist
        else:
            ws = G.mapping(z, c, update_emas=update_emas)
            ws[:, :self.pose_layers] = w_avg.lerp(ws[:, :self.pose_layers], self.pose_trunc_dist)

        ws_aligned = ws.clone()
        ws_aligned[:, :self.pose_layers] = w_avg.lerp(ws[:, :self.pose_layers], psi)

        ws_input = torch.cat([ws, ws_aligned])

        img_set = G.synthesis(ws_input, update_emas=update_emas)

        img, img_aligned = img_set.chunk(2)
        return img, img_aligned,  ws, ws_aligned


    def run_T(self, img, blur_sigma=0, return_full=False, update_emas=False):
        #if self.augment_pipe is not None:
        #    img = self.augment_pipe(img)

        ret = self.T(img, blur_sigma=blur_sigma, return_full=return_full, update_emas=update_emas)

        if return_full:
            transformed_img, mat, depth = ret
            return transformed_img, mat, depth

        else:
            transformed_img = ret

            return transformed_img


    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        # set up constants
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        
        # cosine anneal, from gangealing.utils.annealing.cosine_anneal
        psi = 0.5 * (1 + torch.cos(torch.tensor(math.pi * min(cur_nimg//1000, self.psi_anneal) / self.psi_anneal)))
        psi = psi.to(self.device)

        training_stats.report('Loss/psi', psi)

        # run G & T
        with torch.autograd.profiler.record_function('Gmain_forward'):
            img, img_aligned,  ws, ws_aligned = self.run_G(gen_z, gen_c, psi=psi)

            transformed_img = self.run_T(img.detach(), return_full=False, blur_sigma=blur_sigma)
            img = torch.cat([transformed_img, img_aligned], dim=0)
            
            lpips_t0, lpips_t1 = self.vgg(img, resize_images=False, return_lpips=True).chunk(2)
            perceptual_loss= (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
            training_stats.report('Loss/perceptual_loss', perceptual_loss)

        with torch.autograd.profiler.record_function('Gmain_backward'):
            perceptual_loss.mean().mul(gain).backward()

            #pass

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
