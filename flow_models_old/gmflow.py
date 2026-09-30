import torch
from thirdparty.unimatch.unimatch import UniMatch

from .optical_flow_model import OpticalFlowModel

class GMFlowOpticalFlowModel(OpticalFlowModel):
    def __init__(self, model_config):
        super().__init__()

        model = UniMatch(feature_channels=128,
                        num_scales=2,
                        upsample_factor=4,
                        num_head=1,
                        ffn_dim_expansion=4,
                        num_transformer_layers=6,
                        task='flow').to("cuda")
        checkpoint = torch.load('pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth', map_location="cuda")
        model.load_state_dict(checkpoint['model'], strict=False)
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

        output = self.model(image1 * 255.0, image2 * 255.0,
                            attn_type='swin',
                            attn_splits_list=[2, 8],
                            corr_radius_list=[-1, 4],
                            prop_radius_list=[-1, 1],
                            num_reg_refine=6,
                            task='flow',
                            pred_bidir_flow=False)
        flow_final = output['flow_preds'][-1] # (B, 2, H, W)

        return flow_final