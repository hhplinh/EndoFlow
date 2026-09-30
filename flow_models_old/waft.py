from thirdparty.WAFT.config.parser import json_to_args
from thirdparty.WAFT.model import fetch_model
from thirdparty.WAFT.utils.utils import load_ckpt
from thirdparty.WAFT.inference_tools import InferenceWrapper, AverageMeter

from .optical_flow_model import OpticalFlowModel

class WAFTOpticalFlowModel(OpticalFlowModel):
    def __init__(self):
        super().__init__()

        # model_config = {
        #     "cfg": "thirdparty/WAFT/config/a1/tar-c-t.json",
        #     "ckpt": "weights/WAFT/a1/tar-c-t.pth",
        #     "scale": 0.0,
        # }

        model_config = {
            "cfg": "thirdparty/WAFT/config/a2/dinov3/tar-c-t.json",
            "ckpt": "weights/WAFT/waftv2-ckpts/dinov3/zero-shot.pth",
            "scale": 0.0,
        }

        args = json_to_args(model_config["cfg"])
        model = fetch_model(args)
        load_ckpt(model, model_config["ckpt"])
        model = model.cuda()
        model.eval()
        wrapped_model = InferenceWrapper(model, scale=model_config["scale"], pad_to_train_size=False, tiling=False)

        self.model = wrapped_model

    def compute_optical_flow(self, image1, image2):
        """
        Args:
            image1: (B, 3, H, W) [0.0, 1.0]
            image2: (B, 3, H, W) [0.0, 1.0]
        Returns:
            flow: (B, 2, H, W)
        """

        output = self.model.calc_flow(image1 * 255.0, image2 * 255.0)
        final_flow = output['flow'][-1] # (B, 2, H, W)

        return final_flow
