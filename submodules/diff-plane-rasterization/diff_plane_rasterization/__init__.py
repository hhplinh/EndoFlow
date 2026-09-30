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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
from ._C import (
    N_IN_MAP,
    N_IN_DATA,
)

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    means2D_abs,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    gs_in_data,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        gs_in_data,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        gs_in_data,
        raster_settings,
    ):
        if raster_settings.debug: print("CUSTOM FWD CALLED", flush=True)
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, #0
            means3D, #1
            colors_precomp, #2
            opacities, #3
            scales, #4
            rotations, #5
            raster_settings.scale_modifier, #6
            cov3Ds_precomp, #7
            raster_settings.viewmatrix, #9
            raster_settings.projmatrix, #10
            raster_settings.tanfovx,#11
            raster_settings.tanfovy,#12
            raster_settings.image_height,#13
            raster_settings.image_width,#14
            sh,#15
            gs_in_data,
            raster_settings.sh_degree,#16
            raster_settings.campos,#17
            raster_settings.prefiltered,#18
            raster_settings.render_geo,#19
            raster_settings.debug#20
        )
        (
            num_rendered,
            color,
            radii,
            out_observe,
            input_all_maps,
            cached_neg_masks,
            out_all_map,
            out_plane_depth,
            output_map,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            out_all_map,
            colors_precomp,
            input_all_maps,
            cached_neg_masks,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            gs_in_data,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        if raster_settings.debug: print("\noutput color.sum() this call", color.sum(), flush=True)
        return (
            color,
            radii,
            out_observe,
            out_all_map,
            out_plane_depth,
            output_map,
        )

    @staticmethod
    def backward(ctx,
                 grad_out_color,
                 grad_radii,
                 grad_out_observe,
                 grad_out_all_map,
                 grad_out_plane_depth,
                 grad_output_map):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        if raster_settings.debug: print("AUTOGRAD CALL THIS CLASS BACKWARD", flush=True)
        (
            all_map_pixels,
            colors_precomp,
            input_all_maps,
            cached_neg_masks,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            gs_in_data,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                all_map_pixels,
                means3D, 
                radii, 
                colors_precomp, 
                input_all_maps,
                cached_neg_masks,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_all_map,
                grad_out_plane_depth,
                sh,
                gs_in_data,
                grad_output_map,
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.render_geo,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        (
            grad_means2D,
            grad_means2D_abs,
            grad_colors_precomp,
            grad_opacities,
            grad_means3D,
            grad_cov3Ds_precomp,
            grad_sh,
            grad_scales,
            grad_rotations,
            grad_all_map,
            grad_gs_in_data,
        ) = _C.rasterize_gaussians_backward(*args)
        # print(f"grad_means2D {grad_means2D.sum()}, grad_means2D_abs {grad_means2D_abs.sum()}")

        grads = (
            grad_means3D,
            grad_means2D,
            grad_means2D_abs,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_gs_in_data,
            None,
        )
        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    render_geo : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self,
                means3D,
                means2D,
                means2D_abs,
                opacities,
                gs_in_data,
                shs = None,
                colors_precomp = None,
                scales = None,
                rotations = None,
                cov3D_precomp = None,
                all_map=None):
        
        raster_settings = self.raster_settings

        if (tuple(gs_in_data.shape) != (means3D.shape[0], N_IN_DATA)):
            raise Exception(f'Input data shape {gs_in_data.shape} does not match expected shape {(means3D.shape[0], N_IN_DATA)}')

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if all_map is None:
            all_map = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            means2D_abs,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            gs_in_data,
            raster_settings, 
        )

def backproject_gaussians(
    input_map,
    mask,
    means3D,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    backprojector_settings,
):
    return _BackProjectGaussians.apply(
        input_map,
        mask,
        means3D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        backprojector_settings,
    )

class _BackProjectGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_map,
        mask,
        means3D,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        backprojector_settings,
    ):
        args = (
            input_map,
            mask,
            means3D,
            opacities,
            scales,
            rotations,
            backprojector_settings.scale_modifier,
            cov3Ds_precomp,
            backprojector_settings.viewmatrix,
            backprojector_settings.projmatrix, 
            backprojector_settings.tanfovx,
            backprojector_settings.tanfovy,
            backprojector_settings.image_height,
            backprojector_settings.image_width,
            backprojector_settings.campos,
            backprojector_settings.prefiltered,
        )

        (
            gs_out_data,
        ) = _C.backproject_gaussians(*args)

        # Keep relevant tensors for backward
        
        return (gs_out_data,)

    @staticmethod
    def backward(ctx, gs_out_data):
        # Restore necessary values from context
        return (
            None,  # input_map
            None,  # mask
            None,  # means3D
            None,  # opacities
            None,  # scales
            None,  # rotations
            None,  # cov3Ds_precomp
            None,  # backprojector_settings
        )

class GaussianBackProjectorSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    campos : torch.Tensor
    prefiltered : bool

class GaussianBackProjector(nn.Module):
    def __init__(self, backprojector_settings):
        super().__init__()
        self.backprojector_settings = backprojector_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            backprojector_settings = self.backprojector_settings
            visible = _C.mark_visible(
                positions,
                backprojector_settings.viewmatrix,
                backprojector_settings.projmatrix)
            
        return visible

    def forward(self, 
                input_map,
                mask,
                means3D,
                opacities,
                scales = None,
                rotations = None,
                cov3D_precomp = None,):
        
        backprojector_settings = self.backprojector_settings

        if (tuple(input_map.shape) != (backprojector_settings.image_height, backprojector_settings.image_width, N_IN_MAP)):
            raise Exception(f'Input map shape {input_map.shape} does not match expected shape {(backprojector_settings.image_height, backprojector_settings.image_width, N_IN_MAP)}')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return backproject_gaussians(
            input_map,
            mask,
            means3D,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            backprojector_settings, 
        )
