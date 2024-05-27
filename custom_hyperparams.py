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

from dataclasses import dataclass, field
from typing import List


@dataclass
class DRiseParams:
    # For D-RISE
    NUMBER_OF_MASKS: int = 6000
    STRIDE: int = 8
    P1: float = 0.8
    GPU_BATCH: int = 64
    GENERATE_NEW_MASKS: bool = False


@dataclass
class XAIParams:

    MAX_IOU_WITH_XAI: float = 0.1  # The maximum IOU between an UNK proposal bbox and a predicted bbox. If over the threshold, the UNK proposal is discarded
    assert MAX_IOU_WITH_XAI > 0 and MAX_IOU_WITH_XAI <= 1, "MAX_IOU_WITH_XAI must be between 0 and 1"

    XAI_METHOD: str = "LayerCAM"  # GradCAM, HiResCAM, LayerCAM, D-RISE
    XAI_TARGET_LAYERS: List[int] = (15, 18, 21)
    XAI_RENORMALIZE: bool = False
    INFO_MERGING_METHOD: str = "multiply"  # scale_then_minus, multiply, turn_off_pixels, sigmoid

    if INFO_MERGING_METHOD == "multiply":
        SIGMOID_SLOPE: float = 8.0  # The slope of the sigmoid, more value means a steeper slope, making pixels will be either low or high
        SIGMOID_INTERCEPT: float = 0.5  # The value at which the sigmoid will be 0.5 (moves the sigmoid to the left or right)
    if XAI_METHOD == "D-RISE":
        drise: DRiseParams = DRiseParams()


@dataclass
class RankParams:
    RANK_BOXES_OPERATION: str = "entropy"  # mean, max, min, median, sum, geometric_mean, harmonic_mean, entropy
    MAX_NUM_UNK_BOXES_PER_IMAGE: int = 3  # The maximum number of UNK proposals that will be considered for the ranking
    GET_BOXES_WITH_GREATER_RANK: bool = False  # If True, the boxes with the greater rank will be selected. If False, the boxes with the lower rank will be selected
    NMS: float = 0.5  # If > 0, the NMS will be applied to the ranked boxes
    USE_OOD_THR_TO_REMOVE_PROPS: bool = False  # If True, the boxes with a lower OOD score than the threshold will be removed, 
    USE_UNK_PROPOSALS_THR: bool = True  # If True, the UNK proposals will be selected using a threshold
    if USE_UNK_PROPOSALS_THR or USE_OOD_THR_TO_REMOVE_PROPS:
        assert MAX_NUM_UNK_BOXES_PER_IMAGE > 0, "MAX_NUM_UNK_BOXES_PER_IMAGE must be greater than 0"
    if USE_OOD_THR_TO_REMOVE_PROPS:
        assert RANK_BOXES_OPERATION == "min", "RANK_BOXES_OPERATION must be 'min' when USE_OOD_THR_TO_REMOVE_PROPS is True"


@dataclass
class UnkEnhancementParams:
    # Information summarization method
    SUMMARIZATION_METHOD: str = "mean_absolute_deviation_of_ftmaps"

    # Thresholding method
    THRESHOLDING_METHOD: str = "recursive_otsu"  # multithreshold_otsu, recursive_otsu, k_means, quantile, fast_otsu
    NUM_THRESHOLDS: int = 3  # The number of thresholds to be used in the thresholding methods
    OTSU_RECURSIVE_TRICK_FOR_4_THRS: bool = True  # If True, the first threshold's value will be removed from the saliency map

    # Enable or disable the use of XAI
    USE_XAI_TO_MODIFY_SALIENCY: bool = False  # If True, the XAI method will be used to enhance the localization of the UNK proposals
    USE_XAI_TO_REMOVE_PROPOSALS: bool = True  # If True, the XAI method will be used to remove the UNK proposals
    assert not (USE_XAI_TO_MODIFY_SALIENCY and USE_XAI_TO_REMOVE_PROPOSALS), "Only one of the XAI methods can be used"

    # Enable or disable the use of heuristics
    USE_HEURISTICS: bool = True
    
    # Used heuristics
    USE_FIRST_THRESHOLD: bool = True  # If True, the first threshold will be used
    MAX_IOU_WITH_PREDS: float = 0.5  # The maximum IOU between an UNK proposal bbox and a predicted bbox. If over the threshold, the UNK proposal is discarded
    MIN_BOX_SIZE: int = 3  # The minimum size of a box in the feature map space. 3*8 = 24 pixels in the original image if the stride is 8.
    MAX_BOX_SIZE_PERCENT: float = 0.9  # The percentage of the feature map size that a box can take
    MAX_INTERSECTION_W_PREDS: float = 0.9  # If True, the proposals containing predicted bboxes will be removed

    # Enable or disable ranking the prosals. if USE_HEURISTICS is False, no ranking will be done
    RANK_BOXES: bool = True
    
    if USE_XAI_TO_MODIFY_SALIENCY or USE_XAI_TO_REMOVE_PROPOSALS:
        USE_XAI: bool = True  # If True, the XAI method will be used to enhance the localization of the UNK proposals
        xai: XAIParams = XAIParams()
    else:
        USE_XAI: bool = False
    if RANK_BOXES:
        rank: RankParams = RankParams()
    

@dataclass
class Hyperparams:
    # For YOLO
    IOU_THRESHOLD: float = 0.5
    # Nested dataclasses for better organization
    unk: UnkEnhancementParams = UnkEnhancementParams()

    # Enable or disable the use of a small subset
    USE_ONLY_SUBSET_OF_IMAGES: bool = True  # If True, only a subset of images will be used for the UNK enhancement

    IMAGES_TO_SELECT: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.USE_ONLY_SUBSET_OF_IMAGES:
            
            self.IMAGES_TO_SELECT = [
                "000000361551", "000000154425", "000000334521", "000000231527", "000000231097",
                "000000500613", "000000365655", "000000241319", "000000357060", "000000163951",
                "000000156292", "000000553221", "000000195918", "000000163746", "000000469192"
            ]

# Create an instance of the Hyperparams class for imports
CUSTOM_HYP = Hyperparams()
