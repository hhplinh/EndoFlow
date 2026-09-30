from .foundation_stereo_model import FoundationStereoDepthModel

DEPTH_MODELS = {
    "foundation_stereo": FoundationStereoDepthModel,
}

__all__ = ["DEPTH_MODELS"]