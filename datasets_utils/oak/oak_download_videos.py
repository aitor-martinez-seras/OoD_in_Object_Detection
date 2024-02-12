from pathlib import Path
from multiprocessing import Pool


import gdown
import pandas as pd


def download_one_file(row):
    global root_path
    # Extract the folder's files in GDrive
    row = row[1]
    file_name = row['names']
    url = row['urls']
    file_id = url.split('/')[-1]
    
    list_of_downloaded_files = [d.name for d in sorted(root_path.iterdir())]

    if file_name in list_of_downloaded_files:
        status = 'OK'
    else:
        status = file_name

    file_path = root_path / file_name
    if status != 'OK':
        #downloaded_file = gdown.download(url=url, output=file_path.as_posix(), quiet=False)
        downloaded_file = gdown.download(id=file_id, output=file_path.as_posix(), quiet=False)
        if downloaded_file == None:
            status = f'{file_name}: NOT OK after download'
    else:
        status = f'{file_name}: OK'
    print(status)
    return status


def download_oak_videos_multiprocessing(split:str, option: str, workers):
    global root_path

    
    if split == 'train':
        root_path = Path('/home/tri110414/nfs_home/datasets/OAK/train')
        videos_path = 'Videos'
        root_path = root_path / videos_path
        root_path.mkdir(exist_ok=True)
        df = pd.read_csv('/home/tri110414/nfs_home/yolo-pruebas/datasets_utils/oak/train_video_urls.csv', names=['names', 'urls'], header=None)

        # Multiprocessing
        with Pool(workers) as p:
            _ = p.map(download_one_file, df.iterrows())

    if split == 'val':
        url_val_videos_folder = 'https://drive.google.com/drive/folders/1KBtuyyj30RjpZkBGlDfEyd1sUb0lr5GO'
        folder_id = url_val_videos_folder.split('/')[-1]
        root_path = Path('/home/tri110414/nfs_home/datasets/OAK/val')
        videos_path = 'Videos'
        root_path = root_path / videos_path
        root_path.mkdir(exist_ok=True)
        gdown.download_folder(url=url_val_videos_folder, output=root_path.as_posix(), quiet=False)

    # url_of_video_folder = {
    #     'names': 'Videos',
    #     'urls': 'https://drive.google.com/drive/folders/1RaC0zPyKOnNlAr_J8T25GsTSlm-lEu5n'
    # }
    # check_and_download_one_folder([0, url_of_video_folder])

    print(f'SUCCESFULLY FINISHED ALL DOWNLOADS OF {option} of {split}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='OAK Dataset downloader', 
        description='Downloads the OAK dataset. It must be restarted until message "SUCCESFULLY FINISHED ALL DOWNLOADS" appears',
    )
    parser.add_argument('-s', '--split', required=True, type=str, help='train, val')
    #parser.add_argument('-opt', '--option', type=str, help='images, labels or videos')
    parser.add_argument('-w', '--workers', type=int, default=64, help='num workers')
    #parser.add_argument('--only_check', action='store_true', help='only check if the files are downloaded')
    #parser.add_argument('-ow', '--overwrite_folders', action='store_true', help='images or labels')
    args = parser.parse_args()
    download_oak_videos_multiprocessing(args.split, 'videos', args.workers)
