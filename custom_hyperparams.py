from dataclasses import dataclass, field
from typing import List, Any, Dict


def hyperparams_to_dict(hyperparams: Any) -> Dict:
    data = {}
    # Helper function to extract fields recursively
    def extract_fields(prefix: str, obj: Any):
        if hasattr(obj, '__dataclass_fields__'):
            for field_name, field_value in obj.__dataclass_fields__.items():
                value = getattr(obj, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    extract_fields(f"{prefix}{field_name}.", value)
                else:
                    data[f"{prefix}{field_name}"] = value

    extract_fields("", hyperparams)
    return data


@dataclass
class IvisParams:
    # For Ivis
    EMBEDDING_DIMS: int = 32
    N_EPOCHS_WITHOUT_PROGRESS: int = 20
    K: int = 15
    MODEL: str = 'maaten'


@dataclass
class DimensionalityReductionParams:
    ivis: IvisParams = IvisParams()


@dataclass
class FusionParams:
    
    # Score fusion strategy
    CLIP_FUSION_SCORES: bool = True  # If True, the fusion scores will be clipped to the range [0, 1]  # True

    LOGITS_USE_PIECEWISE_FUNCTION: bool = True  # If True, the piecewise function will be used to fuse the oodness scores

    DISTANCE_USE_FROM_ZERO_TO_THR: bool = False  # If True, the distance from zero to the threshold will be used to fuse the oodness scores  # False
    DISTANCE_USE_IN_DISTRIBUTION_TO_DEFINE_LIMITS: bool = True  # If True, the distances of the distribution of the In-Distribution will be used to define the limits of the piecewise function  # True

    # Assert only one of the two methods of distance is used
    assert not (DISTANCE_USE_FROM_ZERO_TO_THR and DISTANCE_USE_IN_DISTRIBUTION_TO_DEFINE_LIMITS), "Only one of the two distance methods can be used"


@dataclass
class ClustersParams:
    MIN_SAMPLES: int = 3  # Should be equal to the MIN_NUMBER_OF_SAMPLES_FOR_THR
    RANGE_OF_CLUSTERS: List[int] = field(default_factory=lambda: list(range(2, 15)))
    VISUALIZE: bool = False  # This is activated from ood_evaluation
    # Density based metric usage
    USE_DENSITY_BASED_METRIC: bool = False  # If True, the DBCV metric will be used to optimize the number of clusters for density based methods
    # Orphans options (orphans = samples whose label is -1)
    MAKE_EACH_ORPHAN_EACH_OWN_CLUSTER: bool = False  # If True, each orphan will be considered as a cluster
    REMOVE_ORPHANS: bool = False  # If True, orphans will not be counted as a cluster
    #WEIGHT_SCORE_WITH_PERCENT_ORPHANS: bool = True  # If True, the score will be weighted by the percentage of orphans in the cluster
    MAX_PERCENT_OF_ORPHANS: float = 0.95  # The maximum percentage of orphans per class and stride
    assert (MAKE_EACH_ORPHAN_EACH_OWN_CLUSTER != REMOVE_ORPHANS) or (MAKE_EACH_ORPHAN_EACH_OWN_CLUSTER == REMOVE_ORPHANS == False), "Only one of the two options can be used"


@dataclass
class RankParams:
    RANK_BOXES_OPERATION: str = "entropy"  # mean, max, min, median, sum, geometric_mean, harmonic_mean, entropy

    MAX_NUM_UNK_BOXES_PER_IMAGE: int = 3  # The maximum number of UNK proposals that will be considered for the ranking
    GET_BOXES_WITH_GREATER_RANK: bool = False  # If True, the boxes with the greater rank will be selected. If False, the boxes with the lower rank will be selected

    NMS: float = 0.5  # If > 0, the NMS will be applied to the ranked boxes

    # Used with min operation
    USE_OOD_THR_TO_REMOVE_PROPS: bool = False  # If True, the boxes with a lower OOD score than the threshold will be removed, 

    # Used with all operations
    USE_UNK_PROPOSALS_THR: bool = False  # If True, the UNK proposals will be selected using a threshold

    # Checks
    if USE_UNK_PROPOSALS_THR or USE_OOD_THR_TO_REMOVE_PROPS:
        assert MAX_NUM_UNK_BOXES_PER_IMAGE > 0, "MAX_NUM_UNK_BOXES_PER_IMAGE must be greater than 0"
    if USE_OOD_THR_TO_REMOVE_PROPS:
        assert RANK_BOXES_OPERATION == "min", "RANK_BOXES_OPERATION must be 'min' when USE_OOD_THR_TO_REMOVE_PROPS is True"


@dataclass
class UnkEnhancementParams:
    # Unknown enhancement method being used? This is activated from ood_evaluation
    USE_UNK_ENHANCEMENT: bool = False  # DO NOT CHANGE THIS VALUE, IT IS USED TO FOR REPORTS, DOES NOT AFFECT THE CODE

    # Enable or disable the use of heuristics to remove boxes
    USE_HEURISTICS: bool = True

    # Information summarization method
    SUMMARIZATION_METHOD: str = "mean_absolute_deviation_of_ftmaps"

    # Thresholding method
    THRESHOLDING_METHOD: str = "recursive_otsu"  # multithreshold_otsu, recursive_otsu, k_means, quantile, fast_otsu
    NUM_THRESHOLDS: int = 3  # The number of thresholds to be used in the thresholding methods
    OTSU_RECURSIVE_TRICK_FOR_4_THRS: bool = False  # If True, the first threshold's value will be removed from the saliency map
    
    # Simple heuristics
    USE_SIMPLE_HEURISTICS: bool = False  # Use to enable or disable the simple heuristics, listed below
    USE_FIRST_THRESHOLD: bool = True  # If True, the first threshold will be used
    MIN_BOX_SIZE: int = 1  # The minimum size of a box in the feature map space. 3*8 = 24 pixels in the original image if the stride is 8.
    MAX_BOX_SIZE_PERCENT: float = 0.95  # The percentage of the feature map size that a box can take
    MAX_IOU_WITH_PREDS: float = 0.0  # The maximum IOU between an UNK proposal bbox and a predicted bbox. If over the threshold, the UNK proposal is discarded
    MAX_INTERSECTION_W_PREDS: float = 0.0  # If above 0, the proposals with an intersection with the predicted bboxes over the threshold will be removed

    # Enable or disable ranking the prosals. if USE_HEURISTICS is False, no ranking will be done
    RANK_BOXES: bool = True
    rank: RankParams = RankParams()
    

@dataclass
class Hyperparams:
    # For YOLO
    IOU_THRESHOLD: float = 0.5

    # For thresholds of the OOD methods
    GOOD_NUM_SAMPLES: int = 25
    MIN_NUMBER_OF_SAMPLES_FOR_THR: int = 5
    
    # For the clustering of distance methods
    clusters: ClustersParams = ClustersParams()

    # For the dimensionality reduction methods
    dr: DimensionalityReductionParams = DimensionalityReductionParams()

    # Fusion method
    fusion: FusionParams = FusionParams()

    # Enable or disable the use of a small subset
    USE_ONLY_SUBSET_OF_IMAGES: bool = False  # If True, only a subset of images will be used for the UNK enhancement
    IMAGES_TO_SELECT: List[str] = field(default_factory=list)
    
    unk: UnkEnhancementParams = UnkEnhancementParams()

    def __post_init__(self):
        if self.USE_ONLY_SUBSET_OF_IMAGES:
            
            self.IMAGES_TO_SELECT = [
                "000000361551", "000000154425", "000000334521", "000000231527", "000000231097",
                "000000500613", "000000365655", "000000241319", "000000357060", "000000163951",
                "000000156292", "000000553221", "000000195918", "000000163746", "000000469192"
            ]
    
    BENCHMARK_MODE: bool = False  # This should only be changed to True dynamically when running the benchmarks

# Create an instance of the Hyperparams class for imports
CUSTOM_HYP = Hyperparams()
