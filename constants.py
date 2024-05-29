from pathlib import Path

### Paths ###
ROOT = Path(__file__).parent  # Assumes this script is in the root of the project
STORAGE_PATH = ROOT / 'storage'
PRUEBAS_ROOT_PATH = ROOT / 'pruebas'
RESULTS_PATH = ROOT / 'results'

### For YOLO ###
STRIDES_RATIO = [8, 16, 32]  # The ratio between each level of the FPN and the original image size ((orig_img_size / feature_map_size) = ratio)

### For PLOTS ###
IMAGE_FORMAT = 'jpg'

#### OOD Related ####
UNKNOWN_CLASS_INDEX = 80
# Datasets
COCO_OOD_NAME = 'coco_ood'
COCO_MIXED_NAME = 'coco_mixed'
COCO_OWOD_TEST_NAME = 'coco_owod_test'
# OOD Methods
LOGITS_METHODS = ['MSP', 'Energy', 'ODIN', 'Sigmoid']
DISTANCE_METHODS = ['L1_cl_stride', 'L2_cl_stride', 'GAP_L2_cl_stride', 'Cosine_cl_stride']
OOD_METHOD_CHOICES = LOGITS_METHODS + DISTANCE_METHODS

FTMAPS_RELATED_OPTIONS = ['roi_aligned_ftmaps','all_ftmaps', 'ftmaps_and_strides']
LOGITS_RELATED_OPTIONS = ['logits']
INTERNAL_ACTIVATIONS_EXTRACTION_OPTIONS = FTMAPS_RELATED_OPTIONS + LOGITS_RELATED_OPTIONS

AVAILABLE_CLUSTERING_METHODS = ['one', 'DBSCAN', 'KMeans', 'HDBSCAN', 'AgglomerativeClustering', 'OPTICS', 'Birch', 'MeanShift', 'SpectralClustering', 'OPTICS', 'GMM', 'BGMM']
AVAILABLE_CLUSTER_OPTIMIZATION_METRICS = ['silhouette', 'calinski_harabasz']

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

### Benchmarks ###
COMMON_COLUMNS = ['Method', 'Conf_threshold', 'tpr_thr', 'cluster_method']
VOC_TEST_COLUMN = ['mAP']
COCO_OOD_COLUMNS = ['U-AP_(COOD)', 'U-F1_(COOD)', 'U-PRE_(COOD)', 'U-REC_(COOD)']
COCO_MIX_COLUMNS = ['mAP', 'U-AP_(CMIX)', 'U-F1_(CMIX)', 'U-PRE_(CMIX)', 'U-REC_(CMIX)','A-OSE', 'WI']
COCO_OWOD_COLUMNS = ['mAP', 'U-AP_(OWOD)', 'U-F1_(OWOD)', 'U-PRE_(OWOD)', 'U-REC_(OWOD)','A-OSE_(OWOD)', 'WI_(OWOD)']
# Benchmark options
AVAILABLE_BENCHMARKS = ['best_methods', 'conf_thr_test', 'clusters', 'logits_methods']
BENCHMARKS = {
    'best_methods': OOD_METHOD_CHOICES,
    'conf_thr_test': [0.15, 0.10, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001],
    'clusters': ['one', 'DBSCAN', 'KMeans', 'HDBSCAN', 'AgglomerativeClustering', 'OPTICS', 'Birch', 'MeanShift'],
    'cluster_perf_metric': AVAILABLE_CLUSTER_OPTIMIZATION_METRICS,
    'logits_methods': LOGITS_METHODS,
}
# Benchmark configurations
# CONF_THR_TEST_BENCHMARK = [0.15, 0.10, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]
# ALL_METHODS_BENCHMARK = ['MSP', 'Energy', 'ODIN', 'Sigmoid', 'L1_cl_stride', 'L2_cl_stride', 'Cosine_cl_stride']
# CLUSTER_METHODS_BENCHMARK = ['one', 'DBSCAN', 'KMeans', 'HDBSCAN', 'AgglomerativeClustering', 'OPTICS', 'Birch', 'MeanShift']
