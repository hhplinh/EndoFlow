from .optical_flow import OpticalFlowModel
from .scene_flow import SCENE_FLOW_MODELS
from .depth import DEPTH_MODELS

__all__ = [
    "OpticalFlowModel",
]

def get_scene_flow_model(model_name, *args, **kwargs):
    if model_name not in SCENE_FLOW_MODELS:
        raise ValueError(f"Scene flow model '{model_name}' is not supported. Supported models: {list(SCENE_FLOW_MODELS.keys())}")
    return SCENE_FLOW_MODELS[model_name](*args, **kwargs)

def get_depth_model(model_name, *args, **kwargs):
    if model_name not in DEPTH_MODELS:
        raise ValueError(f"Depth model '{model_name}' is not supported. Supported models: {DEPTH_MODELS}")
    return DEPTH_MODELS[model_name](*args, **kwargs)