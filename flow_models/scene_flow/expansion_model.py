import numpy as np
import torch
import torch.nn.functional as F

from .scene_flow_model import SceneFlowModel

from expansion.models.VCN_exp import (
    VCN,
    WarpModule,
    flow_reg,
)


class ExpansionSceneFlowModel(SceneFlowModel):
    """
    Wrapper for:
        "Upgrading Optical Flow to 3D Scene Flow through Optical Expansion"

    Expected inputs:
        image1, image2:
            [B, 3, H, W], RGB, range [0, 1]

        depth1, depth2:
            [B, 1, H, W], camera-space Z depth

        intrinsics1, intrinsics2:
            [B, 3, 3]

        camera_pose1, camera_pose2:
            [B, 4, 4], camera -> world

    Output:
        scene_flow_world:
            [H, W, 3]

    Notes:
        - Currently assumes B = 1 because the original Expansion
          evaluation code returns logmid only for the first sample.
        - depth2 is not used. Expansion predicts Z2 / Z1 itself.
    """

    def __init__(
        self,
        model_path="weights/expansion/robust/robust.pth",
        maxdisp=256,
        fac=1,
        device="cuda",
    ):
        super().__init__()

        self.device = torch.device(device)

        if self.device.type != "cuda":
            raise RuntimeError(
                "The original Expansion implementation contains hard-coded "
                ".cuda() calls and therefore requires CUDA."
            )

        # Important for get_grid(), which internally calls .cuda()
        if self.device.index is not None:
            torch.cuda.set_device(self.device)

        self.maxdisp = maxdisp
        self.fac = fac

        # ---------------------------------------------------------
        # Construct VCN
        #
        # VCN size convention:
        #     [B, W, H]
        #
        # Official submission.py constructs the sequence model using
        # 1280 x 384 before loading the checkpoint.
        # ---------------------------------------------------------
        self.model = VCN(
            [1, 1280, 384],
            md=[int(4 * (maxdisp / 256)), 4, 4, 4, 4],
            fac=fac,
            exp_unc=("robust" in model_path.lower()),
        )

        # ---------------------------------------------------------
        # Load checkpoint
        # ---------------------------------------------------------
        try:
            checkpoint = torch.load(
                model_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # Older PyTorch versions
            checkpoint = torch.load(
                model_path,
                map_location="cpu",
            )

        state_dict = checkpoint["state_dict"]

        # Official checkpoints were commonly saved from DataParallel.
        # Remove "module." so we don't need DataParallel here.
        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------
        # Image normalization means stored in checkpoint
        # ---------------------------------------------------------
        mean_L = np.asarray(checkpoint["mean_L"]).mean(axis=0)
        mean_R = np.asarray(checkpoint["mean_R"]).mean(axis=0)

        self.mean_L = torch.tensor(
            mean_L,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

        self.mean_R = torch.tensor(
            mean_R,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

    @staticmethod
    def _ceil64(x):
        return ((x + 63) // 64) * 64

    def _resize_internal_modules(
        self,
        height,
        width,
    ):
        """
        VCN contains resolution-dependent flow regression and warp grids.

        The official submission.py rebuilds these modules whenever
        the input resolution changes.
        """

        for i in range(len(self.model.reg_modules)):
            scale = 2 ** (6 - i)
            base_reg = getattr(self.model, f"flow_reg{scale}")

            self.model.reg_modules[i] = flow_reg(
                [1, width // scale, height // scale],
                ent=base_reg.ent,
                maxdisp=base_reg.md,
                fac=base_reg.fac,
            ).to(self.device)

        for i in range(len(self.model.warp_modules)):
            # warp_modules[0] corresponds to scale 64 but is None
            # in the original architecture.
            if i == 0:
                continue
            scale = 2 ** (6 - i)
            self.model.warp_modules[i] = WarpModule([1, width // scale, height // scale]).to(self.device)

    def prepare_input(
        self,
        image1,
        image2,
    ):
        """
        Args:
            image1: [1, 3, H, W], RGB [0, 1]
            image2: [1, 3, H, W], RGB [0, 1]

        Returns:
            img_pair: [2, 3, H_model, W_model]
            target_h
            target_w
        """

        assert image1.ndim == 4
        assert image2.ndim == 4
        assert image1.shape == image2.shape

        B, C, orig_h, orig_w = image1.shape

        assert B == 1
        assert C == 3

        target_h = self._ceil64(orig_h)
        target_w = self._ceil64(orig_w)

        image1 = image1.to(device=self.device, dtype=torch.float32)
        image2 = image2.to(device=self.device, dtype=torch.float32)

        # ---------------------------------------------------------
        # Resize exactly as submission.py
        # ---------------------------------------------------------
        if (target_h, target_w) != (orig_h, orig_w):
            image1 = F.interpolate(image1, size=(target_h, target_w), mode="bilinear", align_corners=False)
            image2 = F.interpolate(image2, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # ---------------------------------------------------------
        # RGB -> BGR
        #
        # Official submission.py ultimately feeds BGR into VCN:
        #
        # cv2.imread()       -> BGR
        # [:,:,::-1]         -> RGB
        # ...
        # [:,:,::-1]         -> BGR before network
        # ---------------------------------------------------------
        image1 = image1[:, [2, 1, 0]]
        image2 = image2[:, [2, 1, 0]]

        # ---------------------------------------------------------
        # Checkpoint-specific mean subtraction
        # ---------------------------------------------------------
        image1 = image1 - self.mean_L
        image2 = image2 - self.mean_R

        # VCN convention:
        #
        #     [image1_batch,
        #      image2_batch]
        #
        # For B=1:
        #
        #     [2, 3, H, W]
        # ---------------------------------------------------------
        img_pair = torch.cat([image1, image2], dim=0)
        self._resize_internal_modules(target_h, target_w)

        return img_pair, target_h, target_w

    @staticmethod
    def _backproject(
        depth,
        intrinsics,
    ):
        """
        Back-project camera-space Z depth.

        Args:
            depth:
                [B, 1, H, W]

            intrinsics:
                [B, 3, 3]

        Returns:
            xyz_camera:
                [B, H, W, 3]
        """

        B, _, H, W = depth.shape

        device = depth.device
        dtype = depth.dtype

        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        ones = torch.ones_like(x)
        pixels = torch.stack([x, y, ones], dim=-1)

        # [B, H, W, 3]
        pixels = pixels.unsqueeze(0).expand(B, -1, -1, -1)
        K_inv = torch.linalg.inv(intrinsics)
        rays = torch.einsum("bij,bhwj->bhwi", K_inv, pixels)
        depth = depth.permute(0, 2, 3, 1)
        xyz_camera = rays * depth

        return xyz_camera

    @staticmethod
    def _backproject_correspondence(
        flow,
        depth2,
        intrinsics2,
    ):
        """
        Back-project the corresponding points in image 2.

        Args:
            flow:
                [B, 2, H, W]
                optical flow image1 -> image2

            depth2:
                [B, 1, H, W]
                Z2 associated with each image-1 pixel

            intrinsics2:
                [B, 3, 3]

        Returns:
            xyz2_cam2:
                [B, H, W, 3]
        """

        B, _, H, W = flow.shape

        device = flow.device
        dtype = flow.dtype

        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )

        u2 = x[None] + flow[:, 0]
        v2 = y[None] + flow[:, 1]

        ones = torch.ones_like(u2)

        pixels2 = torch.stack([u2, v2, ones], dim=-1)
        K2_inv = torch.linalg.inv(intrinsics2)
        rays2 = torch.einsum("bij,bhwj->bhwi", K2_inv, pixels2)
        xyz2_cam2 = rays2 * depth2.permute(0, 2, 3, 1)

        return xyz2_cam2

    @staticmethod
    def _camera_to_world(
        xyz_camera,
        camera_pose,
    ):
        """
        Args:
            xyz_camera:
                [B, H, W, 3]

            camera_pose:
                [B, 4, 4]
                camera -> world

        Returns:
            xyz_world:
                [B, H, W, 3]
        """

        R = camera_pose[:, :3, :3]
        t = camera_pose[:, :3, 3]

        xyz_world = torch.einsum("bij,bhwj->bhwi", R, xyz_camera) + t[:, None, None, :]

        return xyz_world

    @torch.no_grad()
    def compute_scene_flow(
        self,
        image1,
        image2,
        depth1,
        depth2,
        intrinsics1,
        intrinsics2,
        camera_pose1,
        camera_pose2,
    ):
        """
        Args:
            image1:
                (B, 3, H, W), RGB [0.0, 1.0]

            image2:
                (B, 3, H, W), RGB [0.0, 1.0]

            depth1:
                (B, 1, H, W), camera-space Z depth

            depth2:
                (B, 1, H, W)
                NOT USED.
                Kept only for compatibility with SceneFlowModel.

            intrinsics1:
                (B, 3, 3)

            intrinsics2:
                (B, 3, 3)

            camera_pose1:
                (B, 4, 4), camera -> world

            camera_pose2:
                (B, 4, 4), camera -> world

        Returns:
            scene_flow_world:
                (H, W, 3)
        """
        B, _, orig_h, orig_w = image1.shape

        if B != 1:
            raise ValueError("ExpansionSceneFlowModel currently supports B=1 only.")

        # ---------------------------------------------------------
        # Move geometry to correct device
        # ---------------------------------------------------------
        depth1 = depth1.to(device=self.device, dtype=torch.float32)

        intrinsics1 = intrinsics1.to(device=self.device, dtype=torch.float32)
        intrinsics2 = intrinsics2.to(device=self.device, dtype=torch.float32)
        camera_pose1 = camera_pose1.to(device=self.device, dtype=torch.float32)
        camera_pose2 = camera_pose2.to(device=self.device, dtype=torch.float32)

        if depth1.ndim == 3:
            depth1 = depth1.unsqueeze(1)

        # ---------------------------------------------------------
        # Prepare images for VCN
        # ---------------------------------------------------------
        img_pair, model_h, model_w = self.prepare_input(image1, image2)

        # ---------------------------------------------------------
        # Expansion inference
        #
        # flow:
        #     [1, 2, model_h, model_w]
        #
        # logmid:
        #     [model_h, model_w]
        #
        # logmid = log(Z2 / Z1)
        # ---------------------------------------------------------
        self.model.eval()

        flow, _, logmid, _ = self.model(img_pair)

        # ---------------------------------------------------------
        # Restore optical flow to original resolution
        # ---------------------------------------------------------
        flow = F.interpolate(flow, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        flow[:, 0] *= orig_w / model_w
        flow[:, 1] *= orig_h / model_h

        # ---------------------------------------------------------
        # Restore log depth ratio
        # ---------------------------------------------------------
        logmid = logmid[None, None]
        logmid = F.interpolate(logmid, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        # ---------------------------------------------------------
        # Expansion prediction:
        #
        # logmid = log(Z2 / Z1)
        #
        # therefore:
        #
        # Z2 = Z1 * exp(logmid)
        #
        # Note:
        # depth2_pred is indexed by pixels of IMAGE 1.
        # It is the depth of the corresponding point after motion.
        # ---------------------------------------------------------
        depth2_pred = depth1 * torch.exp(logmid)

        # ---------------------------------------------------------
        # X1:
        #
        #     (u, v, Z1)
        #       |
        #       v
        #     camera 1 XYZ
        # ---------------------------------------------------------
        xyz1_cam1 = self._backproject(depth1, intrinsics1)

        # ---------------------------------------------------------
        # X2:
        #
        #     (u + flow_x,
        #      v + flow_y,
        #      Z2)
        #
        #       |
        #       v
        #
        #     camera 2 XYZ
        # ---------------------------------------------------------
        xyz2_cam2 = self._backproject_correspondence(flow, depth2_pred, intrinsics2)

        # ---------------------------------------------------------
        # Camera coordinates -> common world coordinates
        # ---------------------------------------------------------
        xyz1_world = self._camera_to_world(xyz1_cam1, camera_pose1)
        xyz2_world = self._camera_to_world(xyz2_cam2, camera_pose2)

        # ---------------------------------------------------------
        # Physical scene flow
        # ---------------------------------------------------------
        scene_flow_world = xyz2_world - xyz1_world

        return scene_flow_world[0]