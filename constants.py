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
COCO_OWOD_TEST_NAME = 'owod'
# OOD Methods
LOGITS_METHODS = ['NoMethod', 'MSP', 'Energy', 'ODIN', 'Sigmoid']
DISTANCE_METHODS = ['L1_cl_stride', 'L2_cl_stride', 'GAP_L2_cl_stride', 'Cosine_cl_stride']
OOD_METHOD_CHOICES = LOGITS_METHODS + DISTANCE_METHODS

FTMAPS_RELATED_OPTIONS = ['roi_aligned_ftmaps','all_ftmaps', 'ftmaps_and_strides']
LOGITS_RELATED_OPTIONS = ['logits']
INTERNAL_ACTIVATIONS_EXTRACTION_OPTIONS = FTMAPS_RELATED_OPTIONS + LOGITS_RELATED_OPTIONS + ['none']  # None for fusion methods, that implement it internally

AVAILABLE_CLUSTERING_METHODS = ['one', 'all', 'DBSCAN', 'KMeans', 'KMeans_4', 'KMeans_10', 'HDBSCAN', 'AgglomerativeClustering', 'OPTICS', 'Birch', 'MeanShift', 'SpectralClustering', 'OPTICS', 'GMM', 'BGMM']
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
COMMON_COLUMNS = ['Method', 'which_split', 'conf_thr_train', 'conf_thr_test', 'tpr_thr', 'cluster_method', 'mean_n_clus', 'std_n_clus',
                  #'mean_num_samples_per_clus', 'std_num_samples_per_clus',
                  'fusion_strat']
VOC_TEST_COLUMN = ['mAP']
COCO_OOD_COLUMNS = ['U-AP_(COOD)', 'U-F1_(COOD)', 'U-PRE_(COOD)', 'U-REC_(COOD)']
COCO_MIX_COLUMNS = ['mAP', 'U-AP_(CMIX)', 'U-F1_(CMIX)', 'U-PRE_(CMIX)', 'U-REC_(CMIX)','A-OSE', 'WI-08']
COCO_OWOD_COLUMNS = ['mAP_(OWOD)', 'U-AP_(OWOD)', 'U-F1_(OWOD)', 'U-PRE_(OWOD)', 'U-REC_(OWOD)','A-OSE_(OWOD)', 'WI-08_(OWOD)']
FINAL_COLUMNS = ['Model', 'args', 'custom_hyp']
# Benchmark options
AVAILABLE_BENCHMARKS = ['best_methods', 'conf_thr_test', 'clusters', 'logits_methods']
BENCHMARKS = {
    'best_methods': OOD_METHOD_CHOICES,
    'used_tpr': [0.99, 0.95, 0.90, 0.85, 0.80],
    'conf_thr_train': [0.50, 0.40, 0.35, 0.25, 0.15, 0.05, 0.01, 0.001],
    'conf_thr_test': [0.15, 0.10, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001],
    'which_split_for_ind_scores': ['train', 'val', 'train_val'],
    'cluster_methods': ['one', 'all', 'DBSCAN', 'KMeans', 'KMeans_4', 'KMeans_10', 'HDBSCAN', 'AgglomerativeClustering', 'Birch'],
    'cluster_perf_metric': AVAILABLE_CLUSTER_OPTIMIZATION_METRICS,
    'logits_methods': LOGITS_METHODS,
    'fusion_strategies': [['fusion-MSP-Energy', 'fusion-MSP-Cosine_cl_stride', 'fusion-Cosine_cl_stride-Cosine_cl_stride'], ['and', 'or', 'score']]
}
# Benchmark configurations
# CONF_THR_TEST_BENCHMARK = [0.15, 0.10, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]
# ALL_METHODS_BENCHMARK = ['MSP', 'Energy', 'ODIN', 'Sigmoid', 'L1_cl_stride', 'L2_cl_stride', 'Cosine_cl_stride']
# CLUSTER_METHODS_BENCHMARK = ['one', 'DBSCAN', 'KMeans', 'HDBSCAN', 'AgglomerativeClustering', 'OPTICS', 'Birch', 'MeanShift']
