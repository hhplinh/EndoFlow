import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

def load_image(image_path):
    image = np.array(Image.open(image_path)).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).cuda().float()
    return image

def calculate_camera_flow(depth1, cam1, cam2, eps=1e-8):
    """
    depth1: [H, W], camera-space Z depth

    Returns:
        camera_flow: [H, W, 2]
    """

    H, W = depth1.shape
    device = depth1.device
    dtype = depth1.dtype

    # ---------------------------------------------------------
    # Pixel coordinates in image 1
    # ---------------------------------------------------------
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )

    uv1 = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

    uv1_h = torch.cat(
        [uv1, torch.ones_like(uv1[..., :1])],
        dim=-1
    )  # [H, W, 3]

    # Row-vector convention:
    # view_image_transform = K^T
    # world_view_transform = T_cw^T
    K1_row = cam1.view_image_transform.to(device=device, dtype=dtype)
    K2_row = cam2.view_image_transform.to(device=device, dtype=dtype)

    E1_row = cam1.world_view_transform.to(device=device, dtype=dtype)
    E2_row = cam2.world_view_transform.to(device=device, dtype=dtype)

    # ---------------------------------------------------------
    # Image 1 -> Camera 1
    # ---------------------------------------------------------
    ray1 = uv1_h @ torch.linalg.inv(K1_row)  # [H, W, 3]

    # Enforce ray Z = 1
    ray1 = ray1 / (ray1[..., 2:3] + eps)

    z1 = depth1.unsqueeze(-1)  # [H, W, 1]

    xyz_cam1 = ray1 * z1  # [H, W, 3]

    xyz_cam1_h = torch.cat(
        [xyz_cam1, torch.ones_like(z1)],
        dim=-1
    )  # [H, W, 4]

    # ---------------------------------------------------------
    # Camera 1 -> World -> Camera 2
    # ---------------------------------------------------------
    xyz_world_h = xyz_cam1_h @ torch.linalg.inv(E1_row)
    xyz_cam2_h = xyz_world_h @ E2_row

    xyz_cam2 = xyz_cam2_h[..., :3]

    # ---------------------------------------------------------
    # Camera 2 -> Image 2
    # ---------------------------------------------------------
    uv2_h = xyz_cam2 @ K2_row

    uv2 = uv2_h[..., :2] / (uv2_h[..., 2:3] + eps)

    # ---------------------------------------------------------
    # Camera-induced optical flow
    # ---------------------------------------------------------
    camera_flow = uv2 - uv1  # [H, W, 2]

    return camera_flow

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ

@torch.no_grad()
def compute_scene_flow_world(
    depth1, depth2,               # [B, 1, H, W], camera-space Z depth
    fwd_flow, bwd_flow,           # [B, 2, H, W]
    mask1, mask2,                 # [B, 1, H, W], True = valid
    K1, K2,                       # [B, 3, 3] or [3, 3]
    T_cw1, T_cw2,                 # [B, 4, 4] or [4, 4], world -> camera
    alpha=0.01,
    beta=0.5,
    min_depth=1e-6,
    max_depth=None,
    remove_occ=True,
    reject_motion_outliers=True,
    outlier_mad_scale=8.0,
):
    """
    Returns:
        scene_flow: [B, 3, H, W], world-space displacement t1 -> t2
        valid:      [B, 1, H, W], True where scene flow is valid
    """
    B, _, H, W = depth1.shape
    device = depth1.device
    dtype = depth1.dtype

    mask1 = mask1.to(device=device).bool()
    mask2 = mask2.to(device=device).bool()

    if mask1.ndim == 3:
        mask1 = mask1[:, None]
    if mask2.ndim == 3:
        mask2 = mask2[:, None]

    if K1.ndim == 2:
        K1 = K1[None].expand(B, -1, -1)
    if K2.ndim == 2:
        K2 = K2[None].expand(B, -1, -1)

    if T_cw1.ndim == 2:
        T_cw1 = T_cw1[None].expand(B, -1, -1)
    if T_cw2.ndim == 2:
        T_cw2 = T_cw2[None].expand(B, -1, -1)

    K1 = K1.to(device=device, dtype=dtype)
    K2 = K2.to(device=device, dtype=dtype)
    T_cw1 = T_cw1.to(device=device, dtype=dtype)
    T_cw2 = T_cw2.to(device=device, dtype=dtype)

    fwd_flow = fwd_flow.to(device=device, dtype=dtype)
    bwd_flow = bwd_flow.to(device=device, dtype=dtype)

    # Image-1 pixel coordinates and their image-2 correspondences.
    uv1 = coords_grid(B, H, W, device=device).to(dtype)
    uv2 = uv1 + fwd_flow

    # Nearest sampling avoids averaging foreground/background depths.
    depth2_warped, target_inside = bilinear_sample(
        depth2,
        uv2,
        mode="nearest",
        return_mask=True,
    )

    mask2_warped = bilinear_sample(
        mask2.to(dtype),
        uv2,
        mode="nearest",
    ) > 0.5

    # Sample backward flow at the actual corresponding positions.
    warped_bwd_flow, bwd_inside = bilinear_sample(
        bwd_flow,
        uv2,
        mode="bilinear",
        return_mask=True,
    )

    # Forward-backward consistency:
    # f12(x) + f21(x + f12(x)) should be approximately zero.
    fb_error = torch.linalg.vector_norm(
        fwd_flow + warped_bwd_flow,
        dim=1,
        keepdim=True,
    )

    flow_magnitude = (
        torch.linalg.vector_norm(fwd_flow, dim=1, keepdim=True)
        + torch.linalg.vector_norm(warped_bwd_flow, dim=1, keepdim=True)
    )

    fb_threshold = alpha * flow_magnitude + beta
    flow_consistent = fb_error <= fb_threshold

    finite = (
        torch.isfinite(depth1)
        & torch.isfinite(depth2_warped)
        & torch.isfinite(fwd_flow).all(dim=1, keepdim=True)
        & torch.isfinite(warped_bwd_flow).all(dim=1, keepdim=True)
    )

    depth_valid = (
        (depth1 > min_depth)
        & (depth2_warped > min_depth)
    )

    if max_depth is not None:
        depth_valid = (
            depth_valid
            & (depth1 < max_depth)
            & (depth2_warped < max_depth)
        )

    valid = (
        target_inside[:, None]
        & bwd_inside[:, None]
        & flow_consistent
        & mask1
        & mask2_warped
        & depth_valid
        & finite
    )

    def unproject_world(depth, uv, K, T_cw):
        N = H * W

        uv_h = torch.cat(
            [
                uv,
                torch.ones(
                    B, 1, H, W,
                    device=device,
                    dtype=dtype,
                ),
            ],
            dim=1,
        ).reshape(B, 3, N)

        rays_camera = torch.linalg.solve(K, uv_h)
        points_camera = rays_camera * depth.reshape(B, 1, N)

        points_camera_h = torch.cat(
            [
                points_camera,
                torch.ones(
                    B, 1, N,
                    device=device,
                    dtype=dtype,
                ),
            ],
            dim=1,
        )

        T_wc = torch.linalg.inv(T_cw)
        points_world_h = torch.bmm(T_wc, points_camera_h)

        return points_world_h[:, :3].reshape(B, 3, H, W)

    points1_world = unproject_world(depth1, uv1, K1, T_cw1)
    points2_world = unproject_world(depth2_warped, uv2, K2, T_cw2)

    scene_flow = points2_world - points1_world

    valid = (
        valid
        & torch.isfinite(points1_world).all(dim=1, keepdim=True)
        & torch.isfinite(points2_world).all(dim=1, keepdim=True)
        & torch.isfinite(scene_flow).all(dim=1, keepdim=True)
    )

    # Optional robust removal of isolated, abnormally large 3D vectors.
    #
    # This should be a final safety filter, not the primary occlusion method.
    # A large genuinely moving object can also have large scene flow.
    if reject_motion_outliers:
        motion_magnitude = torch.linalg.vector_norm(
            scene_flow,
            dim=1,
            keepdim=True,
        )

        for batch_idx in range(B):
            values = motion_magnitude[batch_idx, 0][valid[batch_idx, 0]]

            if values.numel() < 32:
                continue

            median = values.median()
            mad = (values - median).abs().median()

            # Prevent a zero MAD from producing an overly strict threshold.
            robust_sigma = 1.4826 * mad
            scale_floor = 0.05 * median.abs() + 1e-8
            robust_sigma = torch.maximum(robust_sigma, scale_floor)

            maximum_motion = (
                median + outlier_mad_scale * robust_sigma
            )

            valid[batch_idx, 0] &= (
                motion_magnitude[batch_idx, 0] <= maximum_motion
            )

    if remove_occ:
        scene_flow = torch.where(
            valid.expand_as(scene_flow),
            scene_flow,
            torch.zeros_like(scene_flow),
        )

    return scene_flow, valid

@torch.no_grad()
def scene_flow_to_optical_flow(
    scene_flow_world,   # [B, 3, H, W], displacement from t1 to t2
    depth1,             # [B, 1, H, W], camera-1 Z depth
    K1, K2,             # [B, 3, 3] or [3, 3]
    T_cw1, T_cw2,       # [B, 4, 4] or [4, 4], world -> camera
    visible=None,      # [B, 1, H, W], True = valid
    remove_occ=True,
):
    """
    Convert world-space scene flow into optical flow from image 1 to image 2.

    Returns:
        optical_flow: [B, 2, H, W], pixel displacement (dx, dy)
        invalid_mask: [B, 1, H, W], True for invalid/occluded pixels
    """
    B, _, H, W = scene_flow_world.shape
    device, dtype = scene_flow_world.device, scene_flow_world.dtype

    if K1.ndim == 2:
        K1 = K1[None].expand(B, -1, -1)
    if K2.ndim == 2:
        K2 = K2[None].expand(B, -1, -1)
    if T_cw1.ndim == 2:
        T_cw1 = T_cw1[None].expand(B, -1, -1)
    if T_cw2.ndim == 2:
        T_cw2 = T_cw2[None].expand(B, -1, -1)

    K1 = K1.to(device=device, dtype=dtype)
    K2 = K2.to(device=device, dtype=dtype)
    T_cw1 = T_cw1.to(device=device, dtype=dtype)
    T_cw2 = T_cw2.to(device=device, dtype=dtype)

    # Source pixel coordinates.
    uv1 = coords_grid(B, H, W, device=device).to(dtype)  # [B, 2, H, W]
    ones = torch.ones(B, 1, H, W, device=device, dtype=dtype)

    uv1_h = torch.cat([uv1, ones], dim=1).reshape(B, 3, -1)

    # Unproject image-1 pixels into camera-1 coordinates.
    rays1_camera = torch.bmm(torch.linalg.inv(K1), uv1_h)
    points1_camera = rays1_camera * depth1.reshape(B, 1, -1)

    points1_camera_h = torch.cat([points1_camera, torch.ones(B, 1, H * W, device=device, dtype=dtype)], dim=1)

    # Camera 1 -> world.
    T_wc1 = torch.linalg.inv(T_cw1)
    points1_world = torch.bmm(T_wc1, points1_camera_h)[:, :3]

    # Apply world-space scene flow.
    points2_world = points1_world + scene_flow_world.reshape(B, 3, -1)

    points2_world_h = torch.cat([points2_world, torch.ones(B, 1, H * W, device=device, dtype=dtype)], dim=1)

    # World -> camera 2.
    points2_camera = torch.bmm(T_cw2, points2_world_h)[:, :3]
    z2 = points2_camera[:, 2:3]

    # Project into image 2.
    projected2 = torch.bmm(K2, points2_camera)
    uv2 = projected2[:, :2] / z2.clamp_min(1e-8)
    uv2 = uv2.reshape(B, 2, H, W)

    optical_flow = uv2 - uv1

    # finite = (
    #     torch.isfinite(depth1)
    #     & torch.isfinite(scene_flow_world).all(dim=1, keepdim=True)
    #     & torch.isfinite(optical_flow).all(dim=1, keepdim=True)
    # )

    inside = (
        (uv2[:, 0:1] >= 0)
        & (uv2[:, 0:1] <= W - 1)
        & (uv2[:, 1:2] >= 0)
        & (uv2[:, 1:2] <= H - 1)
    )

    valid = (
        (depth1 > 0)
        & (z2.reshape(B, 1, H, W) > 0)
        & inside
        # & finite
    )

    if visible is not None:
        valid = valid & visible.bool()

    if remove_occ:
        optical_flow = torch.where(valid.expand_as(optical_flow), optical_flow, torch.zeros_like(optical_flow))

    return optical_flow, valid

class SceneFlowEstimator:
    def __init__(self):
        self.cache = {}

    def forward(self, 
                fwd_flow,
                bwd_flow,
                depth1,
                depth2,
                mask1,
                mask2,
                K1,
                K2,
                T_cw1,
                T_cw2,
                id1=None,
                id2=None):
        """
        Args:
            fwd_flow: [H, W, 2]
            bwd_flow: [H, W, 2]
            depth1: [H, W]
            depth2: [H, W]
            mask1: [H, W]
            mask2: [H, W]
            K1: [3, 3]
            K2: [3, 3]
            T_cw1: [4, 4]
            T_cw2: [4, 4]
            id1: optional, unique identifier for the first frame
            id2: optional, unique identifier for the second frame
        Returns:
            scene_flow: [H, W, 3]
            valid: [H, W]
        """
        
        if id1 is not None and id2 is not None:
            cache_key = (id1, id2)
            if cache_key in self.cache:
                return self.cache[cache_key]
            
        scene_flow, valid = compute_scene_flow_world(
            depth1=depth1[None, None],
            depth2=depth2[None, None],
            fwd_flow=fwd_flow.permute(2, 0, 1)[None],
            bwd_flow=bwd_flow.permute(2, 0, 1)[None],
            mask1=mask1[None, None].bool(),
            mask2=mask2[None, None].bool(),
            K1=K1[None],
            K2=K2[None],
            T_cw1=T_cw1[None],
            T_cw2=T_cw2[None],
        )

        scene_flow = scene_flow.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
        valid = valid.squeeze(0).squeeze(0)  # (H, W)

        if id1 is not None and id2 is not None:
            self.cache[cache_key] = (scene_flow, valid)

        return scene_flow, valid

    def forward_w_camera(self, fwd_flow, bwd_flow, cam1, cam2, id1=None, id2=None):
        if id1 is not None and id2 is not None:
            cache_key = (id1, id2)
            if cache_key in self.cache:
                return self.cache[cache_key]

        scene_flow, valid = compute_scene_flow_world(
            depth1=cam1.fs_depth.cuda().float()[None, None],
            depth2=cam2.fs_depth.cuda().float()[None, None],
            fwd_flow=fwd_flow.permute(2, 0, 1)[None],
            bwd_flow=bwd_flow.permute(2, 0, 1)[None],
            mask1=cam1.mask.cuda().bool()[None],
            mask2=cam2.mask.cuda().bool()[None],
            K1=cam1.view_image_transform.cuda().transpose(0, 1)[None],
            K2=cam2.view_image_transform.cuda().transpose(0, 1)[None],
            T_cw1=cam1.world_view_transform.cuda().transpose(0, 1)[None],
            T_cw2=cam2.world_view_transform.cuda().transpose(0, 1)[None],
        )

        scene_flow = scene_flow.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
        valid = valid.squeeze(0).squeeze(0)  # (H, W)

        if id1 is not None and id2 is not None:
            self.cache[cache_key] = (scene_flow, valid)

        return scene_flow, valid