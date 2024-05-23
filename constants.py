from pathlib import Path

ROOT = Path(__file__).parent  # Assumes this script is in the root of the project
STORAGE_PATH = ROOT / 'storage'
PRUEBAS_ROOT_PATH = ROOT / 'pruebas'
RESULTS_PATH = ROOT / 'results'

# OOD Related
OOD_METHOD_CHOICES = ['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','React', 'L1_cl_stride', 'L2_cl_stride', \
                      'GAP_L2_cl_stride', 'Cosine_cl_stride']

FTMAPS_RELATED_OPTIONS = ['roi_aligned_ftmaps','all_ftmaps', 'ftmaps_and_strides']
LOGITS_RELATED_OPTIONS = ['logits']
INTERNAL_ACTIVATIONS_EXTRACTION_OPTIONS = FTMAPS_RELATED_OPTIONS + LOGITS_RELATED_OPTIONS

AVAILABLE_CLUSTERING_METHODS = ['one','all','DBSCAN', 'KMeans', 'GMM', 'HDBSCAN', 'OPTICS', 'SpectralClustering', 'AgglomerativeClustering']
AVAILABLE_CLUSTER_OPTIMIZATION_METRICS = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

TARGETS_RELATED_OPTIONS = [
    'all_targets_one_stride',  # Extract the activations from all the targets and only one stride (selected using the bbox size)
    'all_targets_all_strides'  # Extract the activations from all the targets and all the strides
]
PREDICTIONS_RELATED_OPTIONS = [  # Valid predictions = correctly predicted bboxes asociated univocally with one GT bbox
    'valid_preds_one_stride',  # Only the valid predictions and the stride they came from
    'valid_preds_all_strides',  # Only the valid predictions and extract the activations from all the strides
    'all_preds_all_strides',  # Extract the activations from all the predictions (no matter well or bad predicted) and all the strides
]

IND_INFO_CREATION_OPTIONS = TARGETS_RELATED_OPTIONS + PREDICTIONS_RELATED_OPTIONS

# Benchmarks
CONF_THRS_FOR_BENCHMARK = [0.15, 0.10, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]

STRIDES_RATIO = [8, 16, 32]  # The ratio between each level of the FPN and the original image size ((orig_img_size / feature_map_size) = ratio)

# For PLOTS
IMAGE_FORMAT = 'jpg'

### Hyperparameters ###

# # For YOLO
# IOU_THRESHOLD = 0.5

# # For Unknown Localization Enhancement
# MAX_IOU_WITH_PREDS = 0.5  # The maximum IOU between an UNK proposal bbox and a predicted bbox. If over the threshold, the UNK proposal is discarded
# MIN_BOX_SIZE = 3  # The minimum size of a box in the feature map space. 3*8 = 24 pixels in the original image if the stride is 8.
# MAX_BOX_SIZE_PERCENT = 0.9  # The percentage of the feature map size that a box can take
# USE_XAI = True  # If True, the XAI method will be used to enhance the localization of the UNK proposals
# RANK_BOXES = True  # If True, the boxes will be ranked using the OOD method

# # For XAI
# XAI_METHOD = "D-RISE"  # GradCAM, HiResCAM, LayerCAM, D-RISE
# XAI_TARGET_LAYERS = [15, 18, 21]
# XAI_RENORMALIZE = False

# # For Ranking boxes
# RANK_BOXES_OPERATION = "min"  # mean, max, min, median, sum, geometric_mean, harmonic_mean
# MAX_NUM_UNK_BOXES_PER_IMAGE = 5  # The maximum number of UNK proposals that will be considered for the ranking
# GET_BOXES_WITH_GREATER_RANK = False  # If True, the boxes with the greater rank will be selected. If False, the boxes with the lower rank will be selected
