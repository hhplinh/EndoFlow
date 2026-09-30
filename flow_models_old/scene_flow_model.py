import torch
from scene.cameras import Camera

class SceneFlowModel:
    def __init__(self):
        self.cache = {}

    def compute_scene_flow(self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        depth1: torch.Tensor,
        depth2: torch.Tensor,
        intrinsics1: torch.Tensor,
        intrinsics2: torch.Tensor,
        camera_pose1: torch.Tensor,
        camera_pose2: torch.Tensor):

        """
        Args:
            image1: (B, 3, H, W) [0.0, 1.0]
            image2: (B, 3, H, W) [0.0, 1.0]
            depth1: (B, 1, H, W)
            depth2: (B, 1, H, W)
            intrinsics1: (B, 3, 3) (Column-major order)
            intrinsics2: (B, 3, 3) (Column-major order)
            camera_pose1: (B, 4, 4) (Column-major order)
            camera_pose2: (B, 4, 4) (Column-major order)
        Returns:
            flow: (B, 3, H, W) (3D scene flow in world coordinates)
        """

        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        depth1: torch.Tensor,
        depth2: torch.Tensor,
        intrinsics1: torch.Tensor,
        intrinsics2: torch.Tensor,
        camera_pose1: torch.Tensor,
        camera_pose2: torch.Tensor,
        idx1=None,
        idx2=None):

        """
        Args:
            image1: (B, 3, H, W) [0.0, 1.0]
            image2: (B, 3, H, W) [0.0, 1.0]
            depth1: (B, 1, H, W)
            depth2: (B, 1, H, W)
            intrinsics1: (B, 3, 3) (Column-major order)
            intrinsics2: (B, 3, 3) (Column-major order)
            camera_pose1: (B, 4, 4) (Column-major order)
            camera_pose2: (B, 4, 4) (Column-major order)
            idx1: int
            idx2: int
        Returns:
            flow: (B, 3, H, W) (3D scene flow in world coordinates)
        """

        if idx1 is not None and idx2 is not None:
            key = (idx1, idx2)
            if key in self.cache:
                return self.cache[key]

        # Compute scene flow between image1 and image2
        flow = self.compute_scene_flow(
            image1=image1,
            image2=image2,
            depth1=depth1,
            depth2=depth2,
            intrinsics1=intrinsics1,
            intrinsics2=intrinsics2,
            camera_pose1=camera_pose1,
            camera_pose2=camera_pose2)

        if idx1 is not None and idx2 is not None:
            self.cache[(idx1, idx2)] = flow

        return flow

    def forward_w_camera(self,
        camera1: Camera,
        camera2: Camera,
        idx1=None,
        idx2=None):

        """
        Args:
            camera1: Camera object
            camera2: Camera object
            idx1: int
            idx2: int
        Returns:
            flow: (B, 3, H, W) (3D scene flow in world coordinates)
        """

        return self.forward(
            image1=camera1.original_image[None].cuda().float(),
            image2=camera2.original_image[None].cuda().float(),
            depth1=camera1.fs_depth[None].cuda(),
            depth2=camera2.fs_depth[None].cuda(),
            intrinsics1=camera1.view_image_transform[None].cuda().transpose(1, 2).contiguous(),
            intrinsics2=camera2.view_image_transform[None].cuda().transpose(1, 2).contiguous(),
            camera_pose1=camera1.view_world_transform[None].cuda().transpose(1, 2).contiguous(),
            camera_pose2=camera2.view_world_transform[None].cuda().transpose(1, 2).contiguous(),
            idx1=idx1,
            idx2=idx2)