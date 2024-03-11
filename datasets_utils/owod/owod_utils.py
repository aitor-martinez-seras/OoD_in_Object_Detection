from pathlib import Path

DATASETS_FOLDER_PATH = Path(__file__).parents[3] / 'datasets'
COCO_FOLDER_PATH = DATASETS_FOLDER_PATH / 'coco'
PASCAL_VOC_FOLDER_PATH = DATASETS_FOLDER_PATH / 'VOC'
OWOD_FOLDER_PATH = DATASETS_FOLDER_PATH / 'OWOD'


def pascal_voc_split_txt_files_creation():
    # Create the label files in ultralytics format (.txt)
    PASCAL_VOC_SPLIT_FOLDERS = {
        'train': ['train2012', 'train2007', 'val2012', 'val2007'],
        'val': ['test2007'],
        'test': ['test2007']
    }
    voc_images_path = PASCAL_VOC_FOLDER_PATH / 'images'
    for split, folders in PASCAL_VOC_SPLIT_FOLDERS.items():
        with open(PASCAL_VOC_FOLDER_PATH / f'{split}.txt', 'w') as labels_txt:
            for folder in voc_images_path.iterdir():
                if folder.name in folders:
                    for file in folder.iterdir():
                        labels_txt.write(f'./{file.relative_to(PASCAL_VOC_FOLDER_PATH).as_posix()}\n')
    print('Finished PASCAL VOC split .txt files creation')

def owod_split_and_tasks_txt_creation():
    OWOD_SPLIT_FILES = {
        'train': ['VOC/train.txt', 'coco/train2017.txt'],
        'val': ['VOC/val.txt', 'coco/val2017.txt'],
        'test': ['coco/test-dev2017.txt']
    }
    for split, files in OWOD_SPLIT_FILES.items():
        # Create the split file
        with open(OWOD_FOLDER_PATH / f'{split}.txt', 'w') as labels_txt:
            # Now for each split, create the paths file using the paths from the other datasets
            for file in files:
                with open(DATASETS_FOLDER_PATH / file, 'r') as dataset_paths_txt:
                    # The paths must be relative to the datasets root folder
                    # so we need to add the dataset folder to the first part of the path
                    current_dataset = file.split('/')[0]
                    labels_txt.writelines(['./' + current_dataset + img_path[1:] for img_path in dataset_paths_txt.readlines()])   
    print('Finished OWOD split .txt files creation')


if __name__ == '__main__':
    #pascal_voc_split_txt_files_creation()
    owod_split_and_tasks_txt_creation()