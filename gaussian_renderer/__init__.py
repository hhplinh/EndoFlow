#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_plane_rasterization import (
    GaussianRasterizationSettings as PlaneGaussianRasterizationSettings,
    GaussianRasterizer as PlaneGaussianRasterizer,
    GaussianBackProjectorSettings,
    GaussianBackProjector,
    N_IN_MAP,
    N_IN_DATA,
)

from scene.flexible_deform_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image
from typing import List

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    w = viewpoint_cam.image_width
    h = viewpoint_cam.image_height
    fx = w / (2.0 * math.tan(viewpoint_cam.FoVx / 2.0))  # focal length x
    fy = h / (2.0 * math.tan(viewpoint_cam.FoVy / 2.0))  # focal length y
    cx = w / 2.0
    cy = h / 2.0
    
    intrinsic_matrix = torch.tensor([
        [fx/scale,  0,        cx/scale],
        [ 0      , fy/scale,  cy/scale],
        [ 0      ,  0,               1]
    ]).float()
    extrinsic_matrix = viewpoint_cam.world_view_transform.transpose(0,1).contiguous() # cam2world
    # return intrinsic_matrix, extrinsic_matrix
    # intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, gs_in_data: torch.Tensor=None, scaling_modifier = 1.0, override_color = None, override_means3D = None,
           return_plane =True, return_depth_normal = True):

    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # seem like they initial 2Dmeans with zeros and set it to retain grad, 
    # send the its pointer to the CUDA rasterization kernel, 
    # and the CUDA rasterization kernel update/store the projection in this 2Dmeans so that we can inspect the gradient in python env if needed

    if gs_in_data is None:
        gs_in_data = torch.zeros((pc.get_xyz.shape[0], N_IN_DATA), device="cuda")

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    # means3D_finals = []
    # rotations_finals = []
    # scales_finals = []
    # opacity_finals = []
    # means2Ds = []
    # means2D_abss = []
    # shss = []
    # means3D = pc.get_xyz
    # add deformation to each points
    # for pc in pcs:
    # pc = pcs[0]
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0 # [N, ]
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass
    # deformation = pc.get_deformation
    means3D = pc.get_xyz
    # ori_time = torch.tensor(pc.get_normalized_time_with_offset(viewpoint_camera.time)).to(means3D.device)
    ori_time = torch.tensor(viewpoint_camera.time).to(means3D.device)
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None


    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling 
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    means3D_deform, scales_deform, rotations_deform = pc.deformation(means3D[deformation_point], scales[deformation_point], 
                                                                         rotations[deformation_point],
                                                                         ori_time)
    # opacity_final = pc.get_deform_opacity(ori_time)
    use_deform_opacity = False
    if use_deform_opacity:
        opacity_deform = pc.get_deform_opacity(ori_time)
    else:
        opacity_deform = pc._opacity
        
    # print(time.max())
    with torch.no_grad():
        pc._deformation_accum[deformation_point] += torch.abs(means3D_deform - means3D[deformation_point])

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]

    scales_final = pc.scaling_activation(scales_final)
    # print("\n\n\n\n", scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    # opacity_final = pc.opacity_activation(opacity)
    opacity_final = pc.opacity_activation(opacity_final)

    # means3D_final = means3D_final.cuda()
    # rotations_final = rotations_final.cuda()
    # scales_final = scales_final.cuda()
    # opacity_final = opacity_final.cuda()

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    use_deform_color = False
    if use_deform_color:
        shs_deform = pc.get_deform_sh(ori_time)
        shs_final = torch.zeros_like(pc.get_features)
        shs_final[deformation_point] = shs_deform[deformation_point]
        shs_final[~deformation_point] = pc.get_features[~deformation_point]
    else:
        shs_final = shs

    if override_means3D is not None:
        means3D_final = override_means3D
    
    rendered_image, radii, out_observe, out_all_map, plane_depth, output_map = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs_final,
            colors_precomp = colors_precomp,
            opacities = opacity_final,
            scales = scales_final,
            rotations = rotations_final,
            # all_map = input_all_map,
            cov3D_precomp = cov3D_precomp,
            gs_in_data = gs_in_data)

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]
    
    # 1) Compute per-pixel normal magnitude
    #    norm shape: (1, H, W)
    # normal_norm = (rendered_normal ** 2).sum(dim=0, keepdim=True).sqrt()
    # print(normal_norm.mean())

    # rendered_normal = rendered_normal/normal_norm
    # plane_depth = plane_depth/normal_norm

    if return_plane:
        flow_map = output_map[0:2] / (rendered_alpha.detach() + 1e-8)
    else:
        flow_map = output_map[0:2]

    return_dict =  {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "viewspace_points_abs": screenspace_points_abs,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "out_observe": out_observe,
        "rendered_normal": rendered_normal,
        "plane_depth": plane_depth,
        "rendered_distance": rendered_distance,
        "scales_final": scales_final,
        "opacity_final": opacity_final,
        "flow_map": flow_map,
    }
    
    if return_depth_normal:
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        # print((depth_normal ** 2).sum(dim=0, keepdim=True).sqrt().mean())
        return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict

def backproject(viewpoint_camera, pc, pipe, input_map=None, mask=None, scaling_modifier = 1.0):
    if input_map is None:
        input_map = torch.zeros((viewpoint_camera.image_height, viewpoint_camera.image_width, N_IN_MAP), device="cuda")
    if mask is None:
        mask = torch.zeros((viewpoint_camera.image_height, viewpoint_camera.image_width), dtype=torch.bool, device="cuda")

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    backprojector_settings = GaussianBackProjectorSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
        )

    backprojector = GaussianBackProjector(backprojector_settings=backprojector_settings)

    means3D = pc.get_xyz
    # ori_time = torch.tensor(pc.get_normalized_time_with_offset(viewpoint_camera.time)).to(means3D.device)
    ori_time = torch.tensor(viewpoint_camera.time).to(means3D.device)
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling 
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    means3D_deform, scales_deform, rotations_deform = pc.deformation(means3D[deformation_point], scales[deformation_point], 
                                                                         rotations[deformation_point],
                                                                         ori_time)
    # opacity_final = pc.get_deform_opacity(ori_time)
    use_deform_opacity = False
    if use_deform_opacity:
        opacity_deform = pc.get_deform_opacity(ori_time)
    else:
        opacity_deform = pc._opacity

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]

    scales_final = pc.scaling_activation(scales_final)
    # print("\n\n\n\n", scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    # opacity_final = pc.opacity_activation(opacity)
    opacity_final = pc.opacity_activation(opacity_final)

    (gs_out_data, ) = backprojector(
            input_map = input_map,
            mask = mask,
            means3D = means3D_final,
            opacities = opacity_final,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)

    gs_out_data = gs_out_data.detach()
    gs_raw_flow = gs_out_data[:, 0:3]
    gs_weight_sum = gs_out_data[:, 3:4]
    gs_flow = gs_raw_flow / (gs_weight_sum + 1e-8)
    gs_means_2d = gs_out_data[:, 4:6]
    gs_valid = gs_out_data[:, 6:7]

    gs_visible = gs_weight_sum[:, 0] > 0.0 # (N, )

    return_dict =  {
        "gs_flow": gs_flow,
        "gs_means_2d": gs_means_2d,
        "gs_valid": gs_valid,
        "gs_means_3d": means3D_final,
        "gs_visible": gs_visible,
    }
    
    return return_dict
