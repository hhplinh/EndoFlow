from thirdparty.flowseek.utils.utils import json_to_args, load_ckpt
from thirdparty.flowseek import FlowSeek

from .optical_flow_model import OpticalFlowModel

class FlowSeekOpticalFlowModel(OpticalFlowModel):
    def __init__(self):
        super().__init__()

        flowseek_config = {
            "name": "flowseek-M",
            "pretrain": "resnet34",
            "initial_dim": 64,
            "block_dims": [64, 128, 256],
            "radius": 4,
            "dim": 128,
            "num_blocks": 2,
            "iters": 12,
            "da_size": "vitb",
            ###
            "scale": 0,
            "model": "weights/flowseek_M_TartanCT.pth"
        }
        flow_args = json_to_args(flowseek_config)
        model = FlowSeek(flow_args)
        load_ckpt(model, flow_args.model)
        model = model.to("cuda")
        model.eval()

        self.model = model

    def compute_optical_flow(self, image1, image2):
        """
        Args:
            image1: (B, 3, H, W) [0.0, 1.0]
            image2: (B, 3, H, W) [0.0, 1.0]
        Returns:
            flow: (B, 2, H, W)
        """

        output = self.model(image1 * 255.0, image2 * 255.0, iters=12, test_mode=True)
        flow_final = output['flow'][-1] # (B, 2, H, W)

        return flow_final