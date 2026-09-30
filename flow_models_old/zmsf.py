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
        model_path = f"thirdparty/zmsf/configs/{model_config}.yaml"

        with open(model_path, 'r') as f:
            model_config = yaml.safe_load(f)
            
        print(f"Loading model from config: {model_path}")
        self.model = instantiate_component(model_config)
        self.model.to("cuda")

    def prepare_input(self, image, depth, K, P, target_size=(384, 512)):
        """
        image: [1, 3, H, W], RGB images
        depth: [1, H, W], camera-space Z depth
        K: [1, 3, 3], camera intrinsics (column-major)
        P: [1, 4, 4], camera pose (column-major, world -> camera)
        target_size: (H, W), target size for resizing images and depth maps
        """

        # print(f"{image.shape = }, {image.dtype = }, {depth.shape = }, {depth.dtype = }, {K.shape = }, {K.dtype = }, {P.shape = }, {P.dtype = }")

        orig_height, orig_width = image.shape[2], image.shape[3]
        image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)
        depth = F.interpolate(depth.unsqueeze(1), size=target_size, mode='nearest').squeeze(1)

        return {
            'img': image,
            'camera_pose': P,
            'camera_intrinsics': K,
            'depth': depth,
            'dataset': 'demo',
            'instance': '1',
            'orig_size': (orig_width, orig_height)
        }

    def restore_to_original_resolution(self, scene_flow, orig_size):
        """
        Restore dense ZeroMSF outputs to the exact original resolution.

        Args:
            scene_flow:
                Bx3xHxW camera-space 3D offset map.
            orig_size:
                Tuple (original_width, original_height).

        Returns:
            scene_flow_original:
                Bx3xH_originalxW_original.
        """
        
        if scene_flow.ndim != 4 or scene_flow.shape[1] != 3:
            raise ValueError(
                f"Expected scene_flow as Bx3xHxW, got {scene_flow.shape}"
            )

        # Scene flow is a camera-space 3D vector field.
        # Do not multiply its X/Y components by pixel resize ratios.
        scene_flow_original = torch.nn.functional.interpolate(
            scene_flow,
            size=orig_size,
            mode="bilinear",
            align_corners=False,
        )

        return scene_flow_original

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

        input1 = self.prepare_input(image1, depth1, intrinsics1, camera_pose1)
        input2 = self.prepare_input(image2, depth2, intrinsics2, camera_pose2)

        input_dict = {
            "view1": input1,
            "view2": input2,
            "camera_space": self.model.camera_space,
            "first_camera": self.model.first_camera,
            "align_scale": False  # Don't scale since we don't have GT
        }

        orig_width, orig_height = input1['orig_size']

        batch = [input1, input2]
        
        output_dict = self.model.forward(input_dict, batch, None)
        scene_flow = self.model.get_sf(batch, output_dict)  # Bx3xHxW
        
        # Rescale to original aspect ratio
        scene_flow = self.restore_to_original_resolution(scene_flow, (orig_height, orig_width)) # [B, 3, H_orig, W_orig]

        # Convert to world-space scene flow
        R_cw1 = input1['camera_pose'][:, :3, :3] # [B, 3, 3]
        t_cw1 = input1['camera_pose'][:, :3, 3:] # [B, 3, 1]

        scene_flow_world = torch.bmm(R_cw1, scene_flow.reshape(scene_flow.shape[0], 3, -1)) + t_cw1
        scene_flow_world = scene_flow_world.reshape(scene_flow.shape[0], 3, scene_flow.shape[2], scene_flow.shape[3])

        return scene_flow_world