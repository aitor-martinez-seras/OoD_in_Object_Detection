from pathlib import Path

### Paths ###
ROOT = Path(__file__).parent  # Assumes this script is in the root of the project
STORAGE_PATH = ROOT / 'storage'
TEMPORAL_STORAGE_PATH = STORAGE_PATH / 'temp'
PRUEBAS_ROOT_PATH = ROOT / 'pruebas'
RESULTS_PATH = ROOT / 'results'
INDIVIDUAL_RESULTS_FILE_PATH = RESULTS_PATH / 'individual_results'

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
AVAILABLE_DATASETS = [COCO_OOD_NAME, COCO_MIXED_NAME, COCO_OWOD_TEST_NAME]
# OOD Methods
LOGITS_METHODS = ['NoMethod', 'MSP', 'Energy', 'ODIN', 'Sigmoid']
DISTANCE_METHODS = ['L1_cl_stride', 'L2_cl_stride', 'Cosine_cl_stride', 'Umap', 'CosineIvis', 'L1Ivis', 'L2Ivis']
OOD_METHOD_CHOICES = LOGITS_METHODS + DISTANCE_METHODS

FTMAPS_RELATED_OPTIONS = ['roi_aligned_ftmaps','all_ftmaps', 'ftmaps_and_strides', 'ftmaps_and_strides_exact_pos']
LOGITS_RELATED_OPTIONS = ['logits']
INTERNAL_ACTIVATIONS_EXTRACTION_OPTIONS = FTMAPS_RELATED_OPTIONS + LOGITS_RELATED_OPTIONS + ['none']  # None for fusion methods, that implement it internally

AVAILABLE_CLUSTERING_METHODS = ['one', 'all', 'DBSCAN', 'KMeans', 'KMeans_3', 'KMeans_5', 'KMeans_10', 'HDBSCAN', 'AgglomerativeClustering', 'OPTICS', 'Birch', 'MeanShift', 'SpectralClustering', 'OPTICS', 'GMM', 'BGMM']
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
COCO_OWOD_COLUMNS_T1 = ['mAP_(VOC_test)']
FINAL_COLUMNS = ['Model', 'args', 'custom_hyp']
# Benchmark options
AVAILABLE_BENCHMARKS = ['best_methods', 'conf_thr_test', 'clusters', 'logits_methods']
BENCHMARKS = {
    'best_methods': OOD_METHOD_CHOICES,
    'used_tpr': [0.99, 0.95, 0.90, 0.85, 0.80],
    'conf_thr_train': [0.50, 0.40, 0.35, 0.25, 0.15, 0.05, 0.01, 0.001],
    'conf_thr_test': [0.45, 0.30, 0.15, 0.10, 0.05, 0.01, 0.005, 0.001],
    'which_split_for_ind_scores': ['train', 'val', 'train_val'],
    'cluster_methods': ['one', 'all', 'DBSCAN', 'KMeans', 'KMeans_3', 'KMeans_5', 'KMeans_10', 'HDBSCAN', 'AgglomerativeClustering', 'Birch'],
    'cluster_perf_metric': AVAILABLE_CLUSTER_OPTIMIZATION_METRICS,
    'logits_methods': LOGITS_METHODS,
    #'fusion_strategies': [['fusion-MSP-Energy', 'fusion-MSP-Cosine_cl_stride', 'fusion-Cosine_cl_stride-Cosine_cl_stride'], ['and', 'or', 'score']],
    'fusion_strategies': [['fusion-MSP-Sigmoid', 'fusion-MSP-CosineIvis', 'fusion-CosineIvis-Cosine_cl_stride'], ['and', 'or', 'score']],
    'unk_loc_enhancement': [{
        # 'unk.THRESHOLDING_METHOD': ['recursive_otsu'],# 'recursive_otsu'],
        # 'unk.NUM_THRESHOLDS': [3],# 3, 4, 5],
        # 'unk.USE_HEURISTICS': [True],
        # 'unk.MIN_BOX_SIZE': [2,3,5],
        # 'unk.MAX_IOU_WITH_PREDS': [0, 0.25, 0.5, 0.75],
        # 'unk.MAX_INTERSECTION_W_PREDS': [0, 0.25, 0.5, 0.75],
        # 'unk.MAX_BOX_SIZE_PERCENT': [0.9],
        # 'unk.USE_XAI_TO_REMOVE_PROPOSALS': [False],# True],
        # 'unk.USE_XAI_TO_MODIFY_SALIENCY': [False],# True],
        # 'unk.RANK_BOXES': [False],# True],
        # 'unk.xai.XAI_METHOD': ['D-RISE'],

        # 'unk.USE_SIMPLE_HEURISTICS': [False, True],
        # 'unk.USE_XAI_TO_REMOVE_PROPOSALS': [False],
        # 'unk.USE_XAI_TO_MODIFY_SALIENCY': [False],# True],
        # 'unk.RANK_BOXES': [False, True],
        # 'unk.THRESHOLDING_METHOD': ['multithreshold_otsu', 'recursive_otsu'],# 'recursive_otsu'],
        # 'unk.NUM_THRESHOLDS': [3, 4],# 3, 4, 5],
        # 'unk.MAX_IOU_WITH_PREDS': [0, 0.5],
        # 'unk.USE_HEURISTICS': [True],

        'unk.USE_HEURISTICS': [True],
        'unk.RANK_BOXES': [True],
        'unk.THRESHOLDING_METHOD': ['recursive_otsu'],# 'multithreshold_otsu', 'recursive_otsu'],
        'unk.NUM_THRESHOLDS': [3],# 3, 4, 5],
        'unk.rank.MAX_NUM_UNK_BOXES_PER_IMAGE': [3, 5, 7],
        'unk.rank.NMS':[0,25, 0.5, 0.75],  # If > 0, the NMS will be applied to the ranked boxes
        'unk.rank.USE_UNK_PROPOSALS_THR': [True, False]  # If > 0, the NMS will be applied to the ranked boxes
    }],
}
