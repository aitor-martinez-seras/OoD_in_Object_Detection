# # For YOLO
# IOU_THRESHOLD = 0.5

# # For Unknown Localization Enhancement
# MAX_IOU_WITH_PREDS = 0.5  # The maximum IOU between an UNK proposal bbox and a predicted bbox. If over the threshold, the UNK proposal is discarded
# MIN_BOX_SIZE = 3  # The minimum size of a box in the feature map space. 3*8 = 24 pixels in the original image if the stride is 8.
# MAX_BOX_SIZE_PERCENT = 0.9  # The percentage of the feature map size that a box can take

# # For XAI
# USE_XAI = True  # If True, the XAI method will be used to enhance the localization of the UNK proposals
# XAI_METHOD = "D-RISE"  # GradCAM, HiResCAM, LayerCAM, D-RISE
# XAI_TARGET_LAYERS = [15, 18, 21]
# XAI_RENORMALIZE = False

# # For Ranking boxes
# RANK_BOXES = True  # If True, the boxes will be ranked using the OOD method
# RANK_BOXES_OPERATION = "min"  # mean, max, min, median, sum, geometric_mean, harmonic_mean
# MAX_NUM_UNK_BOXES_PER_IMAGE = 5  # The maximum number of UNK proposals that will be considered for the ranking
# GET_BOXES_WITH_GREATER_RANK = False  # If True, the boxes with the greater rank will be selected. If False, the boxes with the lower rank will be selected

from dataclasses import dataclass
from typing import List

@dataclass
class XAIParams:
    USE_XAI: bool = False  # If True, the XAI method will be used to enhance the localization of the UNK proposals
    XAI_METHOD: str = "D-RISE"  # GradCAM, HiResCAM, LayerCAM, D-RISE
    XAI_TARGET_LAYERS: List[int] = (15, 18, 21)
    XAI_RENORMALIZE: bool = False

@dataclass
class RankParams:
    RANK_BOXES: bool = True  # If True, the boxes will be ranked using the OOD method
    RANK_BOXES_OPERATION: str = "geometric_mean"  # mean, max, min, median, sum, geometric_mean, harmonic_mean
    MAX_NUM_UNK_BOXES_PER_IMAGE: int = 5  # The maximum number of UNK proposals that will be considered for the ranking
    GET_BOXES_WITH_GREATER_RANK: bool = False  # If True, the boxes with the greater rank will be selected. If False, the boxes with the lower rank will be selected

@dataclass
class UnkEnhancementParams:
    USE_HEURISTICS: bool = False  # If True, the heuristics will be used to filter out the UNK proposals
    # For Unknown Localization Enhancement
    REMOVE_FIRST_THRESHOLD: bool = True  # If True, the first threshold boxes will not be used
    MAX_IOU_WITH_PREDS: float = 0.5  # The maximum IOU between an UNK proposal bbox and a predicted bbox. If over the threshold, the UNK proposal is discarded
    MIN_BOX_SIZE: int = 3  # The minimum size of a box in the feature map space. 3*8 = 24 pixels in the original image if the stride is 8.
    MAX_BOX_SIZE_PERCENT: float = 0.9  # The percentage of the feature map size that a box can take
    
    xai: XAIParams = XAIParams()
    rank: RankParams = RankParams()

@dataclass
class Hyperparams:
    # For YOLO
    IOU_THRESHOLD: float = 0.5
    # Nested dataclasses for better organization
    unk: UnkEnhancementParams = UnkEnhancementParams()

# Create an instance of the Hyperparams class for imports
CUSTOM_HYP = Hyperparams()

