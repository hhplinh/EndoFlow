from .flowseek import FlowSeekOpticalFlowModel
from .gmflow import GMFlowOpticalFlowModel
from .waft import WAFTOpticalFlowModel

from .zmsf import ZeroMonocularSceneFlowModel

optical_flow_models = {
    "flowseek": FlowSeekOpticalFlowModel,
    "gmflow": GMFlowOpticalFlowModel,
    "waft": WAFTOpticalFlowModel,
}

scene_flow_models = {
    "zmsf": ZeroMonocularSceneFlowModel,
}

def get_optical_flow_model(model_name):
    if model_name not in optical_flow_models:
        raise ValueError(f"Unknown optical flow model: {model_name}")
    return optical_flow_models[model_name]()

def get_scene_flow_model(model_name):
    if model_name not in scene_flow_models:
        raise ValueError(f"Unknown scene flow model: {model_name}")
    return scene_flow_models[model_name]()