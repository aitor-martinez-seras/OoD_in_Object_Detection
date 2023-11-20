from pathlib import Path
import shutil
import json
import os

from PIL import Image
import yaml
from tqdm import tqdm


def generate_ultralytics_yolo_annotations(split: str):

    ### 1. Create dataset yaml file in ultralytics format ###

    # First import the classes of oak dataset
    classes_json_path = 'datasets_utils/oak/oak_classes.json'
    classes_dict = json.load(open(classes_json_path))
    classes_dict_inversed = {v: k for k, v in classes_dict.items()}
    
    dataset_dict = {
        'path': '../datasets/OAK/',
        'train': 'train.txt',
        'val': 'val.txt',
        'test': 'val.txt',
        'nc': len(classes_dict),
        'names': classes_dict_inversed
    }
    
    # Save the dictionary as yaml file
    dataset_yaml_path = 'OAK_full.yaml'
    with open(dataset_yaml_path, 'w') as file:
        yaml.dump(dataset_dict, file, sort_keys=False)
    

    ### 2. Create train.txt and val.txt files in ultralytics format ###
    
    # Root path of the dataset
    # Load json file
    root_path = Path("/home/tri110414/nfs_home/datasets/OAK")
    
    # Old paths
    # Train
    old_root_path_train = root_path / "train"
    old_root_path_train_images = old_root_path_train / "Raw"
    old_root_path_train_labels = old_root_path_train / "Labels"
    # Val
    old_root_path_val = root_path / "val"
    old_root_path_val_images = old_root_path_val / "Raw"
    old_root_path_val_labels = old_root_path_val / "Labels"

    # New paths
    # Images
    new_root_path_images = root_path / "images"
    new_root_path_images_train = new_root_path_images / "train"
    new_root_path_images_val = new_root_path_images / "val"
    # Labels
    new_root_path_labels = root_path / "labels"
    new_root_path_labels_train = new_root_path_labels / "train"
    new_root_path_labels_val = new_root_path_labels / "val"

    # Create new folders
    # Images
    new_root_path_images.mkdir(parents=False, exist_ok=True)
    new_root_path_images_train.mkdir(parents=False, exist_ok=True)
    new_root_path_images_val.mkdir(parents=False, exist_ok=True)
    # Labels
    new_root_path_labels.mkdir(parents=False, exist_ok=True)
    new_root_path_labels_train.mkdir(parents=False, exist_ok=True)
    new_root_path_labels_val.mkdir(parents=False, exist_ok=True)

    # TRAIN SET
    if split in ['train', 'both']:
        # For creating the train.txt file and the labels (that are in .txt files)
        #   I only need to iterate the labels folder
        train_paths_txt_file_path = root_path / "train.txt"
        train_image_paths = []
        print('Train set')
        print('Total nº of folders:', len(sorted(old_root_path_train_labels.iterdir())))
        for idx_folder, folder in tqdm(enumerate(sorted(old_root_path_train_labels.iterdir()))):

            for idx_json, json_file in enumerate(sorted(folder.iterdir())):
                
                # Get the video name
                video_name_parts = json_file.stem.split('_')[:-1]
                video_name = f'{video_name_parts[0]}_{video_name_parts[1]}_{video_name_parts[2]}'

                ### Move or copy the images to new destination from the original one and create a .txt with paths ###
                # Check if new folder for each image of the videos exists
                new_path_images_train_one_video = new_root_path_images_train / video_name
                new_path_images_train_one_video.mkdir(parents=False, exist_ok=True)

                # Copy the image to the new destination. The name of the image is the same as the json file
                #   but with .jpg extension. The destination is the new folder for the images of the video
                image_old_path = Path(old_root_path_train_images / folder.name / f'{json_file.stem}.jpg')
                image_new_path = Path(new_path_images_train_one_video / f'{json_file.stem}.jpg')
                shutil.copy(image_old_path, image_new_path)
                train_image_paths.append(os.path.join('.', *image_new_path.parts[-4:]) + '\n')

                ### Create annotation .txt ###
                # Check if new folder for each label file exists
                new_path_labels_train_one_video = new_root_path_labels_train / video_name
                new_path_labels_train_one_video.mkdir(parents=False, exist_ok=True)
                # The yolo ultralytics format is [class_id, x_center, y_center, width, height] normalized
                # As our bboxes are not normalized and are in the format [x1, y1, x2, y2], we have to convert them
                #   to the ultralytics format
                # For normalization, we need to infer the image size from the image
                img = Image.open(image_new_path)
                img_width, img_height = img.size

                # Then, we have to get the annotations from the json file
                with open(json_file) as f:
                    data = json.load(f)
                    one_image_annotations = []
                    for ann in data:
                        # Create the .txt file with the annotations
                        # They must be in normalized cxcywh
                        # x1 and y1 are the top left corner of the bbox
                        x_center = (ann['box2d']['x1'] + ann['box2d']['x2']) / 2
                        y_center = (ann['box2d']['y1'] + ann['box2d']['y2']) / 2
                        width = ann['box2d']['x2'] - ann['box2d']['x1']
                        height = ann['box2d']['y2'] - ann['box2d']['y1']
                        # Append the annotation to the list with f strings and normalizing it
                        one_image_annotations.append(
                            f'{ann["id"]} {x_center/img_width} {y_center/img_height} {width/img_width} {height/img_height}\n'
                        )
                
                with open(new_path_labels_train_one_video / f'{json_file.stem}.txt', 'w') as f:
                    f.writelines(one_image_annotations)

        # Save the train.txt file
        with open(train_paths_txt_file_path, 'w') as f:
            f.writelines(train_image_paths)

    elif split in ['val', 'both']:
        val_paths_txt_file_path = root_path / "val.txt"
        val_image_paths = []
        print('Val set')
        print('Total nº of folders:', len(sorted(old_root_path_val_labels.iterdir())))
        for idx_folder, folder in tqdm(enumerate(sorted(old_root_path_val_labels.iterdir()))):
            # In validation, each folder corresponds to a video
            video_name = folder.name
            # Create new folder for each image and labels of the videos 
            new_path_images_val_one_video = new_root_path_images_val / video_name
            new_path_images_val_one_video.mkdir(parents=False, exist_ok=True)
            new_path_labels_val_one_video = new_root_path_labels_val / video_name
            new_path_labels_val_one_video.mkdir(parents=False, exist_ok=True)

            for idx_json, json_file in enumerate(sorted(folder.iterdir())):
                # Copy image to new destination
                image_old_path = Path(old_root_path_val_images / video_name / f'{json_file.stem}.jpg')
                image_new_path = Path(new_path_images_val_one_video / f'{json_file.stem}.jpg')
                shutil.copy(image_old_path, image_new_path)
                val_image_paths.append(os.path.join('.', *image_new_path.parts[-4:]) + '\n')
                
                ### Create annotation .txt ###
                img = Image.open(image_new_path)
                img_width, img_height = img.size
                # Then, we have to get the annotations from the json file
                with open(json_file) as f:
                    data = json.load(f)
                    one_image_annotations = []
                    for ann in data:
                        # Create the .txt file with the annotations
                        # They must be in normalized cxcywh
                        # x1 and y1 are the top left corner of the bbox
                        x_center = (ann['box2d']['x1'] + ann['box2d']['x2']) / 2
                        y_center = (ann['box2d']['y1'] + ann['box2d']['y2']) / 2
                        width = ann['box2d']['x2'] - ann['box2d']['x1']
                        height = ann['box2d']['y2'] - ann['box2d']['y1']
                        # Append the annotation to the list with f strings and normalizing it
                        one_image_annotations.append(
                            f'{ann["id"]} {x_center/img_width} {y_center/img_height} {width/img_width} {height/img_height}\n'
                        )
                
                with open(new_path_labels_val_one_video / f'{json_file.stem}.txt', 'w') as f:
                    f.writelines(one_image_annotations)
        # Save the val.txt file
        with open(val_paths_txt_file_path, 'w') as f:
            f.writelines(val_image_paths)
        # Save the test.txt file
        test_paths_txt_file_path = root_path / "test.txt"
        with open(test_paths_txt_file_path, 'w') as f:
            f.writelines(val_image_paths)


if __name__ == '__main__':
    generate_ultralytics_yolo_annotations(split='train')


# if video_fname_prev_iteration != '':
#     Path(video_fname_prev_iteration).mkdir(parents=False, exist_ok=True)

# # Check if it has changed, if it has changed, then we have a new video and we have to update the video_id
# if video_fname != video_fname_prev_iteration:
#     video_id += 1
# video_fname_prev_iteration = video_fname

#  # Get the video name and check the video id
# video_fname_parts = json_file.stem.split('_')[:-1]
# video_fname = f'{video_fname_parts[0]}_{video_fname_parts[1]}_{video_fname_parts[2]}.mp4'
# # Check if it has changed, if it has changed, then we have a new video and we have to update the video_id
# if video_fname != video_fname_prev_iteration:
#     video_id += 1
# video_fname_prev_iteration = video_fname
    