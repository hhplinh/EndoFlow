#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
import json
import numpy as np
import random
import os 
import torch
from random import randint
from utils.loss_utils import l1_loss, get_img_grad_weight, masked_TV_loss, TV_loss, AlignedLoss, progressive_frequency_loss
from gaussian_renderer import render, backproject

import sys
from scene import  Scene
from scene.flexible_deform_model import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import FDMHiddenParams as ModelHiddenParams
from utils.timer import Timer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# import lpips
from utils.scene_utils import render_training_image
from flow_models import OpticalFlowModel
from utils.graphics_utils import image2world, world2image

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# import torch

class ViewpointDataset(Dataset):
    def __init__(self, viewpoint_stack):
        # Save the list of viewpoint cameras
        self.viewpoint_stack = viewpoint_stack

    def __len__(self):
        return len(self.viewpoint_stack)

    def __getitem__(self, index):
        # Return the viewpoint at this index
        return self.viewpoint_stack[index]

def median_filter_3x3(x):
    """
    Applies a 3x3 median filter to a 4D tensor [B, C, H, W] using PyTorch.
    - Each channel is filtered independently.
    - Uses F.unfold + median along the 9 values in each 3x3 window.

    Args:
        x (torch.Tensor): [B, C, H, W] on CPU or CUDA.

    Returns:
        torch.Tensor: Filtered output, same shape [B, C, H, W].
    """
    B, C, H, W = x.shape
    
    # 1) Pad by 1 pixel on each side (reflect or replicate to avoid edge issues)
    #    So the 3x3 window is valid for border pixels as well.
    x_padded = F.pad(x, (1, 1, 1, 1), "constant", 0)  # [B, C, H+2, W+2]

    # 2) Extract all 3x3 patches using unfold
    #    Unfold returns shape: [B, C*K*K, H_out * W_out]
    #    where K=3, H_out=H, W_out=W for stride=1.
    patches = F.unfold(x_padded, kernel_size=3, stride=1)  # [B, C*9, H*W]

    # 3) Reshape so the 9 pixels in the 3x3 window sit on a separate dimension
    #    We want shape [B, C, 9, H*W].
    patches = patches.view(B, C, 9, H*W)  # separate the 9 from C

    # 4) Take the median along dim=2 (the 9-pixel dimension)
    median_vals = patches.median(dim=2).values  # [B, C, H*W]

    # 5) Reshape back to [B, C, H, W]
    out = median_vals.view(B, C, H, W)
    return out

def get_depth_grad_weight(depth_map, beta=2.0):
    """
    Compute a gradient-based weight map for a single-channel depth map.
    The returned gradient weight is normalized to [0, 1] and padded with 1.0 
    at the borders, following the same steps as the original get_img_grad_weight.

    Args:
        depth_map (torch.Tensor): Single-channel depth, shape [H, W].
        beta (float): Optional exponent to apply to the gradient map.

    Returns:
        torch.Tensor: Gradient-based weight, shape [H, W].
    """

    # 1) Get the shape.
    #    Must be [H, W].
    if depth_map.ndim != 2:
        raise ValueError(f"Expected 2D depth map [H, W], got shape {depth_map.shape}")
    H, W = depth_map.shape

    # 2) Sample neighboring pixels (skipping the boundary by 1px).
    bottom_point = depth_map[2:H,   1:W-1]  # shape [H-2, W-2]
    top_point    = depth_map[0:H-2, 1:W-1]
    right_point  = depth_map[1:H-1, 2:W]
    left_point   = depth_map[1:H-1, 0:W-2]

    # 3) Compute horizontal (grad_img_x) and vertical (grad_img_y) gradient magnitudes.
    grad_img_x = torch.abs(right_point - left_point)  # [H-2, W-2]
    grad_img_y = torch.abs(top_point - bottom_point)  # [H-2, W-2]

    # 4) Combine them by taking the maximum (same as original).
    #    We'll temporarily add a channel dim so we can use torch.cat + max like the original.
    grad_img_x = grad_img_x.unsqueeze(0)   # [1, H-2, W-2]
    grad_img_y = grad_img_y.unsqueeze(0)   # [1, H-2, W-2]
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)  # [2, H-2, W-2]
    grad_img, _ = torch.max(grad_img, dim=0)               # [H-2, W-2]

    # 5) Normalize to [0, 1].
    min_val = grad_img.min()
    max_val = grad_img.max()
    # Add a small epsilon in the denominator to avoid division by zero
    grad_img = (grad_img - min_val) / (max_val - min_val + 1e-8)

    # 6) Optionally apply an exponent, if beta != 1.0.
    if beta != 1.0:
        grad_img = grad_img ** beta

    # 7) Pad with 1.0 around the boundary, matching the original code's style.
    #    After padding, shape becomes [H, W].
    grad_img = F.pad(grad_img.unsqueeze(0).unsqueeze(0), (1,1,1,1), value=1.0).squeeze()

    return grad_img

def get_pseudo_normal(x, mask):
    
    # x: N, C, H, W
    mask = mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    z = torch.ones_like(diff_x)
    normal = torch.cat([diff_x, -diff_y, -z], dim=1)
    normal = F.normalize(normal, dim=1)
    normal = normal*mask
    # cv2.imwrite('norm.png', ((normal+1)/2*255).squeeze(0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
    # cv2.imwrite('mask.png', (mask*255).squeeze(0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))

    return normal

def get_closest_camera_by_time(cameras, ref_time):
    """
    Given a list of cameras (each with a .time attribute) and a reference time,
    return the camera whose time is closest to ref_time.
    """
    min_diff = float('inf')
    closest_cam = None
    for cam in cameras:
        diff = abs(cam.time - ref_time)
        if diff < min_diff:
            min_diff = diff
            closest_cam = cam
    return closest_cam

def get_gaussians(gaussians_sets, time_idx):
    for gaussians in gaussians_sets:
        if time_idx >= gaussians.min_time_idx and time_idx < gaussians.max_time_idx:
            return gaussians
    print("\n\n\n fail idx: ", time_idx, "\n")

# def get_adaptive_weight(idx, psnr_dict):
    
def compute_ratio_weights_from_psnr(psnr_dict, beta=3.25,
                                    w_min=1, w_max=2, normalize=False):

    errors_dict = {key: 1.0 / (psnr_dict[key].detach().cpu().item() + 1e-8) for key in psnr_dict}
    mean_err = np.mean(list(errors_dict.values()))

    weights = {}
    for key in errors_dict:
        w_raw = 1.0 + beta * ((errors_dict[key] - mean_err) / (mean_err + 1e-8))
        # clamp
        w_clamped = max(w_min, min(w_raw, w_max))
        weights[key] = w_clamped
    
    if normalize:
        avg_w = np.mean(list(weights.values()))
        if avg_w < 1e-8:
            avg_w = 1e-8
        weights = {i: weights[i] / avg_w for i in weights}
    return weights

def scene_reconstruction(flownet, dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians_sets, scene, tb_writer, train_iter, timer,
                         alpha_3d, flow_loss_weight):
    # print(opt.align_loss_from_iter)
    # _align_loss = AlignedLoss().cuda()
    psnr_dict = {}
    weights_dict = {}
    first_iter = 0
    for gaussians in gaussians_sets:
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None

    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    # lpips_model = lpips.LPIPS(net="vgg").cuda()
    """
    cameras.append(Camera(colmap_id=idx, R=R, T=T, FoVx=FovX, FoVy=FovY,image=image, depth=depth, mask=mask, gt_alpha_mask=None,
                    image_name=f"{idx}", uid=idx, data_device=torch.device("cuda"), time=time,
                    Znear=None, Zfar=None, K=self.K, h=self.img_wh[1], w=self.img_wh[0]))
    """
    video_cams = scene.getVideoCameras()
    
    # if not viewpoint_stack:
    #     # same format as video_cams but sampling for training
    #     # list Camera
    #     viewpoint_stack = scene.getTrainCameras()
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    global_view = scene.getTrainCameras().copy()

    idx_list = list(range(len(viewpoint_stack)))

    for iteration in range(first_iter, final_iter+1): 
        for gaussians in gaussians_sets:
            gaussians.update_learning_rate(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 500 == 0:
                gaussians.oneupSHdegree()

        iter_start.record()
        # sampling one idx/time_step in dataset
        # seem like stocastic GD with batch size 1?
        # ???? epoch ????
        # idx = randint(0, len(viewpoint_stack)-1)
        # viewpoint_cams = [viewpoint_stack[idx]]
        view_idx = randint(0, len(viewpoint_stack)-1)

        viewpoint_cams_1 = [viewpoint_stack.pop(view_idx)]
        ori_idx_1 = idx_list.pop(view_idx)
        ori_idx_0 = max(0, ori_idx_1 - 1)
        ori_idx_2 = min(len(global_view)-1, ori_idx_1 + 1)

        viewpoint_cams_0 = [global_view[ori_idx_0]]
        viewpoint_cams_2 = [global_view[ori_idx_2]]

        # viewpoint_cams = [viewpoint_stack[0]]
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            idx_list = list(range(len(viewpoint_stack)))
            weights_dict = compute_ratio_weights_from_psnr(psnr_dict)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        normals = []
        depth_normals = []
      
        scales_final = torch.empty(0)
        
        for (viewpoint_cam_0, viewpoint_cam_1, viewpoint_cam_2) in zip(viewpoint_cams_0, viewpoint_cams_1, viewpoint_cams_2):
            # render: 
            #   (1) extract time_step find deformation using self._deformation to get deformation of that time step t
            #   (2) add 3DGuassians with its deformation (new object)
            #   (3) call CUDA kernel for rasterization via submodules/depth-diff-gaussian-rasterization
            #       rendered_image, radii, depth = rasterizer(...)
            # gaussians = get_gaussians(gaussians_sets, viewpoint_cam.time)
            # print(gaussians)

            gt_image_0 = viewpoint_cam_0.original_image.cuda().float()
            gt_image_1 = viewpoint_cam_1.original_image.cuda().float()
            gt_image_2 = viewpoint_cam_2.original_image.cuda().float()
            mask_0 = viewpoint_cam_0.mask.cuda()
            mask_1 = viewpoint_cam_1.mask.cuda()
            mask_2 = viewpoint_cam_2.mask.cuda()

            with torch.no_grad():
                flow_map_0_to_1 = flownet.forward_with_camera_flow(
                    gt_image_0, gt_image_1, 
                    viewpoint_cam_0, viewpoint_cam_1, 
                    idx1=viewpoint_cam_0.uid, idx2=viewpoint_cam_1.uid
                )
                flow_map_2_to_1 = flownet.forward_with_camera_flow(
                    gt_image_2, gt_image_1,
                    viewpoint_cam_2, viewpoint_cam_1,
                    idx1=viewpoint_cam_2.uid, idx2=viewpoint_cam_1.uid
                )

            input_map_0 = torch.cat([flow_map_0_to_1, torch.zeros_like(flow_map_0_to_1[..., 0:1])], dim=-1)
            input_map_2 = torch.cat([flow_map_2_to_1, torch.zeros_like(flow_map_2_to_1[..., 0:1])], dim=-1)

            backproj_pkg_0 = backproject(viewpoint_cam_0, gaussians_sets[0], pipe, input_map_0, mask_0)
            # backproj_pkg_1 = backproject(viewpoint_cam_1, gaussians_sets[0], pipe, None, None)
            backproj_pkg_2 = backproject(viewpoint_cam_2, gaussians_sets[0], pipe, input_map_2, mask_2)

            gs_flow_0_to_1 = backproj_pkg_0["gs_flow"][:, :2]
            gs_flow_2_to_1 = backproj_pkg_2["gs_flow"][:, :2]

            visible_0 = backproj_pkg_0["gs_visible"]
            visible_2 = backproj_pkg_2["gs_visible"]

            means3D_0, meansNDC_0, means2D_0 = gaussians_sets[0].get_means(viewpoint_cam_1, time=viewpoint_cam_0.time, use_float64=True)
            means3D_1, meansNDC_1, means2D_1 = gaussians_sets[0].get_means(viewpoint_cam_1, use_float64=True)
            means3D_2, meansNDC_2, means2D_2 = gaussians_sets[0].get_means(viewpoint_cam_1, time=viewpoint_cam_2.time, use_float64=True)

            means2D_0_to_1 = means2D_0 + gs_flow_0_to_1
            means2D_2_to_1 = means2D_2 + gs_flow_2_to_1

            z_ndc = meansNDC_1[..., 2:3]

            full_project = viewpoint_cam_1.full_proj_transform.cuda().double()
            W = viewpoint_cam_1.image_width
            H = viewpoint_cam_1.image_height

            means3D_0_to_1 = image2world(means2D_0_to_1, z_ndc, full_project, W, H, eps=1e-12)
            means3D_2_to_1 = image2world(means2D_2_to_1, z_ndc, full_project, W, H, eps=1e-12)

            # view_dir_0 = F.normalize(means3D_0_to_1 - viewpoint_cam_1.camera_center.cuda(), dim=-1)
            # view_dir_2 = F.normalize(means3D_2_to_1 - viewpoint_cam_1.camera_center.cuda(), dim=-1)
            # # depth_noise_std = 0.01
            # depth_noise_std = 0.01
            
            # means3D_0_to_1_w_depth_noise = means3D_0_to_1 + (view_dir_0 * depth_noise_std) * torch.randn_like(means3D_0_to_1[..., 0:1])
            # means3D_2_to_1_w_depth_noise = means3D_2_to_1 + (view_dir_2 * depth_noise_std) * torch.randn_like(means3D_2_to_1[..., 0:1])

            # alpha_3d = 0.25
            means3D_precomp = alpha_3d * means3D_0_to_1 + (1 - 2 * alpha_3d) * means3D_1 + alpha_3d * means3D_2_to_1
            # means3D_precomp = alpha_3d * means3D_0_to_1_w_depth_noise + (1 - 2 * alpha_3d) * means3D_1 + alpha_3d * means3D_2_to_1_w_depth_noise
            means3D_precomp = means3D_precomp.float()

            # gs_means_2d = backproj_pkg_1["gs_means_2d"]
            # gs_means_3d = backproj_pkg_1["gs_means_3d"]
            # gs_valid = backproj_pkg_1["gs_valid"]

            # print(f"gs_valid: {gs_valid.sum() / gs_valid.numel() * 100:.2f}% valid pixels")

            # means2d_error = (means2D_1.detach().float() - gs_means_2d).abs() * gs_valid
            # print(f"{means2d_error.min() = }, {means2d_error.max() = }, {means2d_error.mean() = }")

            # true_means3D_1, _, true_means2D_1 = gaussians_sets[0].get_means(viewpoint_cam_1)
            # true_means2d_error = (true_means2D_1.detach().float() - gs_means_2d).abs() * gs_valid
            # print(f"{true_means2d_error.min() = }, {true_means2d_error.max() = }, {true_means2d_error.mean() = }")

            # true_means3d_error = (true_means3D_1.detach().float() - gs_means_3d).abs()
            # print(f"{true_means3d_error.min() = }, {true_means3d_error.max() = }, {true_means3d_error.mean() = }")

            # test_means2D_1, test_meansNDC_1 = world2image(means3D_1, full_project, W, H, eps=1e-12, return_ndc=True)
            # test_means3D_1 = image2world(test_means2D_1, test_meansNDC_1[..., 2:3], full_project, W, H, eps=1e-12)
            # means3d_error = (means3D_1 - test_means3D_1).abs().detach()
            # print(f"{means3d_error.min() = }, {means3d_error.max() = }, {means3d_error.mean() = }")

            render_pkg = render(viewpoint_cam_1, gaussians_sets[0], pipe, background, override_means3D=means3D_precomp, return_plane=True, return_depth_normal=True)

            image, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
            gt_depth = viewpoint_cam_1.original_depth.cuda().float()
        
            images.append(image.unsqueeze(0))
            depths.append(render_pkg["plane_depth"].unsqueeze(0))
            gt_images.append(gt_image_1.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask_1.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            normals.append(render_pkg["rendered_normal"].unsqueeze(0))
            depth_normals.append(render_pkg["depth_normal"].unsqueeze(0))
            scales_final = render_pkg["scales_final"]
            # seem like it not being used at all <= 2Dmeans projection with thier gradient
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        
        # written like it can be multiple scene trained it one iteration, but actually was one ...
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0).requires_grad_(True)
        depth_tensor = torch.cat(depths, 0)
        gt_image_tensor = torch.cat(gt_images,0)
        gt_depth_tensor = torch.cat(gt_depths, 0)
   
        mask_tensor = torch.cat(masks, 0)

        normal = torch.cat(normals, 0)
        depth_normal = torch.cat(depth_normals, 0)

        if iteration == 2:
            print(gt_depth_tensor.shape, mask_tensor.shape, depth_tensor.shape)
        if opt.clean_noise:
            gt_image_tensor = median_filter_3x3(gt_image_tensor)
        # abs error
        
   
        Ll1 = l1_loss(image_tensor, gt_image_tensor, mask_tensor)
        loss = Ll1.clone()
        
        if (gt_depth_tensor!=0).sum() < 10:
            depth_loss = torch.tensor(0.).cuda()
        else:
            depth_tensor[depth_tensor!=0] = 1 / depth_tensor[depth_tensor!=0]
            gt_depth_tensor[gt_depth_tensor!=0] = 1 / gt_depth_tensor[gt_depth_tensor!=0]

            # abs error but depth
            depth_loss = l1_loss(depth_tensor, gt_depth_tensor, mask_tensor)

        # # scale loss
        scaling_loss = 0
        if visibility_filter.sum() > 0:
            # TODO: change get_scaling to deform_scaling
            # deform = exp(self._scale + sum(basis*weight)) >= 0
            scale = scales_final[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]

            scaling_loss = opt.scale_loss_weight * min_scale_loss.mean()
        # single-view loss
        normal_loss = 0
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            
            color_grad_mask = get_img_grad_weight(gt_image_1)   # shape [H, W]
            depth_grad_mask = get_depth_grad_weight(gt_depth) # shape [H, W]
        
            color_weight = (1.0 - color_grad_mask).clamp(0,1).detach()
            depth_weight = (1.0 - depth_grad_mask).clamp(0,1).detach()
        
            # weighted normal difference
            edge_aware_weight = depth_weight
            # edge_aware_weight = (0.5 * depth_weight + 0.5 * color_weight) ** 1.25
                
            if not opt.wo_image_weight:
                if opt.regularize_geometry_only_mask:
                    normal_loss =  (torch.logical_not(mask_1)  * weight * (
                        edge_aware_weight * ((depth_normal - normal).abs().sum(dim=0))
                    )).mean()
                else:
                    normal_loss =  weight * (
                        edge_aware_weight * ((depth_normal - normal).abs().sum(dim=0))
                    ).mean()
                
            else:
                if opt.regularize_geometry_only_mask:
                    normal_loss = weight * (torch.logical_not(mask_1) * ((depth_normal - normal).abs().sum(dim=0))).mean()
                else:
                    normal_loss = weight * (((depth_normal - normal).abs().sum(dim=0))).mean()

        tv_depth_loss = 0
        if iteration > opt.tv_depth_loss_from_iter:
            weight = opt.tv_depth_loss_weight
            tv_depth_loss = weight * TV_loss(depth_tensor)

        tv_color_loss = 0
        if iteration > opt.tv_color_loss_from_iter:
            weight = opt.tv_color_loss_weight
            tv_color_loss = weight * TV_loss(image_tensor)

        align_loss = 0
        
        if iteration > opt.align_loss_from_iter:
            cur = viewpoint_cam_1.time
          
            next_view =  get_closest_camera_by_time(global_view, cur)
           
            next_render_pkg = render(next_view, gaussians, pipe, background,
                            return_plane=True, return_depth_normal=True)
            weight = opt.align_loss_weight
            align_loss = weight * _align_loss(image_tensor, next_render_pkg["render"])

       
        psnr_ = psnr(image_tensor, gt_image_tensor, mask_tensor).mean().double()
        psnr_dict[viewpoint_cam_1.time] = psnr_
        # combine loss

        fourier_loss = 0

        mu_reg_loss = torch.tensor(0.0).cuda()
        if iteration > opt.mu_reg_from_iter and iteration < opt.iterations - opt.mu_reg_cool_down_iters:
            mu_reg_loss = gaussians_sets[0].compute_mu_peak_regularization(viewpoint_cam_1.time, visibility_filter)

        flow_loss = torch.tensor(0.0).cuda()
        if iteration > 0 and flow_loss_weight > 0:
            # error = 0.5 * (means2D_0_to_1 - means2D_1).square().mean() + 0.5 * (means2D_2_to_1 - means2D_1).square().mean()
            error_0_to_1 = (means2D_0_to_1 - means2D_1)[visible_0].square()
            error_2_to_1 = (means2D_2_to_1 - means2D_1)[visible_2].square()
            error = (error_0_to_1.sum() + error_2_to_1.sum()) / (visible_0.sum() + visible_2.sum() + 1e-8)
            # flow_loss = 0.001 * error
            flow_loss = flow_loss_weight * error
            # flow_loss = 0.0 * error

        loss = opt.color_weight*Ll1 \
             + opt.depth_weight*depth_loss \
             + scaling_loss \
             + tv_depth_loss \
             + tv_color_loss \
             + normal_loss \
             + flow_loss \
             + mu_reg_loss
            # + fourier_loss
        
        if iteration > 0 and iteration%500 == 0:
    
            print("fourier_loss: ", fourier_loss)
            print("Ll1: ", Ll1)
            print("depth_loss: ", depth_loss)
            print("scaling_loss: ", scaling_loss)
            print("normal_loss: ", normal_loss)
            print("tv_depth_loss: ", tv_depth_loss)
            print("tv_color_loss: ", tv_color_loss)
            print("flow_loss: ", flow_loss)
            # print("coefs_l2_reg: ", coefs_l2_reg)
            print("align_loss: ", align_loss)
            print("mu_reg_loss: ", mu_reg_loss) 
        loss.backward()

      
        # seem like it tery to copy the grad out for further use?
        # viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        # for idx in range(0, len(viewspace_point_tensor_list)):
        #     viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Progress bar

            total_point = sum([gaussians._xyz.shape[0] for gaussians in gaussians_sets ])
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background])
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, 'fine')
            timer.start()
            
        
            idx = gaussians_sets[0].get_xyz.shape[0]
            if iteration < opt.densify_until_iter :
                # for gaussians in gaussians_sets:
                # Keep track of max radii in image-space for pruning
                # this occur EVERY ITERATION , it catch the max radii2D for prunning
                mask = ((render_pkg["out_observe"][:idx] > 0) & visibility_filter[:idx])
                gaussians_sets[0].max_radii2D[mask] = (torch.max(gaussians_sets[0].max_radii2D[mask], radii[:idx][mask]))
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"].grad[:idx]
                gaussians_sets[0].add_densification_stats(render_pkg["viewspace_points"].grad[:idx], viewspace_point_tensor_abs, render_pkg["opacity_final"][:idx], visibility_filter[:idx])
                # gaussians_sets[0].add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                # calculate thds for opacity prunning and densification
                # they are linear function that decrease over time. 
                # prune_mask = (self.get_opacity < min_opacity).squeeze(): opacity_threshold decreasing => less prunning over time
                opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                # torch.norm(grads, dim=-1) >= grad_threshold: grad_threshold decreasing => more densify over time
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )
                densify_threshold_abs = opt.densify_grad_threshold_abs_fine_init - iteration*(opt.densify_grad_threshold_abs_fine_init - opt.densify_grad_threshold_abs_after)/(opt.densify_until_iter )  

                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians_sets[0].prune(densify_threshold, densify_threshold_abs, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians_sets[0].densify(densify_threshold, densify_threshold_abs, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("\n\n\nreset opacity\n\n\n")
                    gaussians_sets[0].reset_opacity()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians_sets[0].optimizer.step()
                gaussians_sets[0].optimizer.zero_grad(set_to_none = True)

            if iteration > opt.basis_restart_from_iter and iteration < opt.iterations - opt.basis_restart_cool_down_iters and iteration % opt.basis_restart_interval == 0:
                gaussians_sets[0].restart_basis_image_space(viewpoint_camera=viewpoint_cam_1)


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark,
             alpha_3d, flow_loss_weight):
    tb_writer = prepare_output_and_logger(expname)
    num_frame_per_set = opt.frame_segmented
    # last set got more not divided
    # num_set = int(opt.num_frame/opt.frame_segmented)
    gaussians_sets = []
    for _ in range(1):
        gaussians_sets.append(GaussianModel(dataset.sh_degree, hyper))

    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians_sets, opt.frame_segmented, opt)

    # flownet = get_optical_flow_model("flowseek")
    flownet = OpticalFlowModel("flowseek_m", "tar-c-t")
    # flownet = OpticalFlowModel("flow_anything", "mixed_tskh432")
    # flownet = OpticalFlowModel("flowseek_m", "tar-c-t-tskh")
    # flownet = OpticalFlowModel("dpflow", "sintel")
    # flownet = OpticalFlowModel("gmflow", "things")
    # flownet = OpticalFlowModel('raft', 'things')

    timer.start()
    scene_reconstruction(flownet, dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians_sets, scene, tb_writer, opt.iterations, timer,
                         alpha_3d, flow_loss_weight)

    print("Model trained in {} seconds.".format(timer.get_elapsed_time()))

    return timer

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    
    if tb_writer:
        tb_writer.add_scalar(f'train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'iter_time', elapsed, iteration)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "endonerf/pulling_fdm")
    parser.add_argument("--configs", type=str, default = "arguments/endonerf/default.py")
    parser.add_argument("--alpha_3d", type=float, default = 0.2)
    parser.add_argument("--flow_loss_weight", type=float, default = 0.001)
    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.save_iterations.append(args.iterations)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    timer = training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.extra_mark,
        args.alpha_3d, args.flow_loss_weight)

    peak_bytes = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory used by tensors: {peak_bytes / 1024**2:.2f} MB")

    peak_reserved = torch.cuda.max_memory_reserved()
    print(f"Peak GPU memory reserved: {peak_reserved / 1024**2:.2f} MB")

    with open(os.path.join(args.model_path, "training_report.json"), 'w') as report_f:
        json.dump({
            "training_time_seconds": timer.get_elapsed_time(),
            "peak_gpu_memory_mb": peak_bytes / 1024**2,
            "peak_gpu_memory_reserved_mb": peak_reserved / 1024**2,
        }, report_f)

    # All done
    print("\nTraining complete.")
