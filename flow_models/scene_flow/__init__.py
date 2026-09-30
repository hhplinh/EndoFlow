from .zmsf_model import ZeroMonocularSceneFlowModel
from .expansion_model import ExpansionSceneFlowModel

SCENE_FLOW_MODELS = {
    "zmsf": ZeroMonocularSceneFlowModel,
    "expansion": ExpansionSceneFlowModel,
}

__all__ = ['SCENE_FLOW_MODELS']