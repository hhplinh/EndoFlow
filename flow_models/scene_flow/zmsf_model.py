from typing import Any, Dict
import importlib
import yaml
import torch
import torch.nn.functional as F

from .scene_flow_model import SceneFlowModel

class ZeroMonocularSceneFlowModel(SceneFlowModel):
    def __init__(self):
        super().__init__()

        def instantiate_component(config: Dict[str, Any]) -> Any:
            """Instantiate a component from config dict with class_path and init_args"""
            if 'class_path' in config:
                module_path, class_name = config['class_path'].rsplit('.', 1)
                module = importlib.import_module(module_path)
                class_ = getattr(module, class_name)
                init_args = {k: instantiate_component(v) if isinstance(v, dict) else v 
                            for k, v in config.get('init_args', {}).items()}
                return class_(**init_args)
            return {k: instantiate_component(v) if isinstance(v, dict) else v 
                    for k, v in config.items()}

        model_config="zmsf_shift+dynamicreplica+vkitti+kubric+pointodyssey+spring_adap_241106"
        model_path = f"thirdparty/zmsf/zmsf/configs/{model_config}.yaml"

        with open(model_path, 'r') as f:
            model_config = yaml.safe_load(f)
            
        print(f"Loading model from config: {model_path}")
        self.model = instantiate_component(model_config)
        self.model.to("cuda")

    def prepare_input(
        self,
        image,
        depth,
        K,
        P,
        target_size=(384, 512),
    ):
        """
        image: [B, 3, H, W]
        depth: [B, H, W]
        K:     [B, 3, 3]
        P:     [B, 4, 4], camera -> world
        """

        B, _, orig_h, orig_w = image.shape
        target_h, target_w = target_size

        image = F.interpolate(image, size=target_size, mode="bilinear", align_corners=False)
        depth = F.interpolate(depth.unsqueeze(1), size=target_size, mode="nearest").squeeze(1)

        # Resize intrinsics
        K = K.clone()

        sx = target_w / orig_w
        sy = target_h / orig_h

        K[:, 0, 0] *= sx
        K[:, 0, 2] *= sx

        K[:, 1, 1] *= sy
        K[:, 1, 2] *= sy

        return {
            "img": image,
            "camera_pose": P,
            "camera_intrinsics": K,
            "depth": depth,
            "dataset": "demo",
            "instance": "1",
            "orig_size": (orig_h, orig_w),
        }

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
            image1: (B, 3, H, W) [0.0, 1.0]
            image2: (B, 3, H, W) [0.0, 1.0]
            depth1: (B, 1, H, W)
            depth2: (B, 1, H, W)
            intrinsics1: (B, 3, 3) (Column-major order)
            intrinsics2: (B, 3, 3) (Column-major order)
            camera_pose1: (B, 4, 4) (Column-major order)
            camera_pose2: (B, 4, 4) (Column-major order)
        Returns:
            flow: (H, W, 3) (3D scene flow in world coordinates)
        """

        input1 = self.prepare_input(image1, depth1, intrinsics1, camera_pose1)
        input2 = self.prepare_input(image2, depth2, intrinsics2, camera_pose2)

        input_dict = {
            "view1": input1,
            "view2": input2,
            "camera_space": self.model.camera_space,
            "first_camera": self.model.first_camera,
            "align_scale": False,
        }

        batch = [input1, input2]

        output_dict = self.model.forward(input_dict, batch, None)

        # ZeroMSF camera-space scene flow
        # [B, 3, h, w]
        scene_flow_cam = self.model.get_sf(batch, output_dict)

        # Important: this implementation assumes the official
        # camera_space=True, first_camera=False configuration.
        assert self.model.camera_space
        assert not self.model.first_camera

        # ---------------------------------------------------------
        # Recover both 3D endpoints
        # ---------------------------------------------------------

        # X1 expressed in camera-1 coordinates
        # [B, h, w, 3]
        xyz1_cam1 = output_dict["xyz_1"]

        # ZeroMSF definition:
        # S = X2_cam2 - X1_cam1
        xyz2_cam2 = (
            xyz1_cam1
            + scene_flow_cam.permute(0, 2, 3, 1)
        )

        # ---------------------------------------------------------
        # Camera coordinates -> common world coordinates
        # ---------------------------------------------------------

        R1 = camera_pose1[:, :3, :3]
        t1 = camera_pose1[:, :3, 3]

        R2 = camera_pose2[:, :3, :3]
        t2 = camera_pose2[:, :3, 3]

        xyz1_world = (
            torch.einsum("bij,bhwj->bhwi", R1, xyz1_cam1)
            + t1[:, None, None]
        )

        xyz2_world = (
            torch.einsum("bij,bhwj->bhwi", R2, xyz2_cam2)
            + t2[:, None, None]
        )

        # Actual physical motion in world coordinates
        scene_flow_world = xyz2_world - xyz1_world

        # ---------------------------------------------------------
        # Restore spatial resolution
        # ---------------------------------------------------------

        orig_h, orig_w = input1["orig_size"]

        scene_flow_world = F.interpolate(
            scene_flow_world.permute(0, 3, 1, 2),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        return scene_flow_world[0].permute(1, 2, 0)