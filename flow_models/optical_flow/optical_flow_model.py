import torch
import ptlflow
from utils.flow_utils import calculate_camera_flow

class OpticalFlowModel:
    def __init__(self, model_name: str, ckpt_path: str):
        self.cache = {}

        self.model = ptlflow.get_model(model_name, ckpt_path=ckpt_path)
        self.model.eval()
        self.model.to('cuda')

    def compute_optical_flow(self, image1, image2):
        """
        Args:
            image1: (3, H, W) [0.0, 1.0]
            image2: (3, H, W) [0.0, 1.0]
        Returns:
            flow: (H, W, 2)
        """
        # RGB to BGR
        image1 = torch.flip(image1, dims=[0])
        image2 = torch.flip(image2, dims=[0])

        inputs = {
            "images": torch.cat([
                image1.unsqueeze(0),
                image2.unsqueeze(0)],
                dim=0
            ).unsqueeze(0), # (1, 2, 3, H, W)
        }

        predictions = self.model(inputs)
        flows = predictions['flows'] # (1, 1, 2, H, W)

        return flows.squeeze(0).squeeze(0).permute(1, 2, 0) # (H, W, 2)

    def forward(self, image1, image2, idx1=None, idx2=None):
        """
        Args:
            image1: (3, H, W) [0.0, 1.0]
            image2: (3, H, W) [0.0, 1.0]
            idx1: int
            idx2: int
        Returns:
            flow: (H, W, 2)
        """

        if idx1 is not None and idx2 is not None:
            key = (idx1, idx2)
            if key in self.cache:
                return self.cache[key]
    
        # Compute optical flow between image1 and image2
        flow = self.compute_optical_flow(image1, image2)

        if idx1 is not None and idx2 is not None:
            self.cache[(idx1, idx2)] = flow

        return flow

    def forward_with_camera_flow(self, image1, image2, cam1, cam2, idx1=None, idx2=None):
        """
        Args:
            image1: (3, H, W) [0.0, 1.0]
            image2: (3, H, W) [0.0, 1.0]
            cam1: Camera object for image1
            cam2: Camera object for image2
            idx1: int
            idx2: int
        Returns:
            motion_flow: (H, W, 2)
        """

        if idx1 is not None and idx2 is not None:
            key = (idx1, idx2)
            if key in self.cache:
                return self.cache[key]

        # Compute optical flow between image1 and image2
        flow = self.compute_optical_flow(image1, image2)

        # Compute camera flow based on depth and camera parameters
        camera_flow = calculate_camera_flow(cam1.original_depth.cuda().float(), cam1, cam2)

        # Compute motion flow by subtracting camera flow from optical flow
        motion_flow = flow - camera_flow

        if idx1 is not None and idx2 is not None:
            self.cache[(idx1, idx2)] = motion_flow

        return motion_flow