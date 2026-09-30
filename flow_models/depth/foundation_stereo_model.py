import os
import torch
from omegaconf import OmegaConf

from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.core.foundation_stereo import FoundationStereo

from .stereo_depth_model import DepthModel

class FoundationStereoDepthModel(DepthModel):
    def __init__(self, hiera=False, remove_invisible=False, valid_iters=32):
        super().__init__()
        self.hiera = hiera
        self.remove_invisible = remove_invisible
        self.valid_iters = valid_iters
        self.scale = 1

        ckpt_dir = "weights/foundation_stereo/pretrained_models/23-51-11/model_best_bp2.pth"

        cfg = OmegaConf.load(
            os.path.join(os.path.dirname(ckpt_dir), "cfg.yaml")
        )
        if "vit_size" not in cfg:
            cfg.vit_size = "vitl"

        cfg["hiera"] = self.hiera
        cfg["remove_invisible"] = self.remove_invisible
        cfg["valid_iters"] = self.valid_iters
        cfg["scale"] = self.scale

        model = FoundationStereo(cfg).cuda().eval()
        checkpoint = torch.load(ckpt_dir, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        self.model = model


    def compute_depth(self, image_left, image_right, fx, baseline):
        image_left = image_left * 255.0
        image_right = image_right * 255.0

        height, width = image_left.shape[1:3]

        padder = InputPadder(image_left.shape, divis_by=32, force_square=False)
        image_left, image_right = padder.pad(image_left[None], image_right[None])

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if self.hiera:
                disparity = self.model.run_hierachical(image_left, image_right, iters=self.valid_iters, test_mode=True, small_ratio=0.5)
            else:
                disparity = self.model(image_left, image_right, iters=self.valid_iters, test_mode=True)

        disparity = padder.unpad(disparity.float())
        disparity = disparity.reshape(height, width)
        # disparity = disparity.cpu().numpy().reshape(height, width)

        scaled_fx = fx * self.scale
        # depth = np.zeros_like(disparity, dtype=np.float32)
        depth = torch.zeros_like(disparity, dtype=torch.float32)

        valid = disparity > 0
        if self.remove_invisible:
            # x = np.arange(width)[None, :]
            x = torch.arange(width, device=disparity.device)[None, :]
            valid &= (x - disparity) >= 0

        depth[valid] = scaled_fx * baseline / disparity[valid]
        return depth