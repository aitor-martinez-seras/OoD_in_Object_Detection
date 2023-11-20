from pathlib import Path
from multiprocessing import Pool
import os.path as osp
import json
import sys
import itertools
import re
import warnings
import time

import bs4
import requests
import gdown
import pandas as pd

#CHUNK_SIZE = 512 * 1024  # 512KB
home = osp.expanduser("~")

##############################################################################################################

# Code based on gdown repo: https://github.com/wkentaro/gdown/tree/main

##############################################################################################################


class _GoogleDriveFile(object):
    TYPE_FOLDER = "application/vnd.google-apps.folder"

    def __init__(self, id, name, type, children=None):
        self.id = id
        self.name = name
        self.type = type
        self.children = children if children is not None else []

    def is_folder(self):
        return self.type == self.TYPE_FOLDER


def _get_session(proxy, use_cookies, return_cookies_file=False):
    sess = requests.session()

    sess.headers.update(
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6)"}
    )

    if proxy is not None:
        sess.proxies = {"http": proxy, "https": proxy}
        print("Using proxy:", proxy, file=sys.stderr)

    # Load cookies if exists
    cookies_file = osp.join(home, ".cache/gdown/cookies.json")
    if osp.exists(cookies_file) and use_cookies:
        with open(cookies_file) as f:
            cookies = json.load(f)
        for k, v in cookies:
            sess.cookies[k] = v

    if return_cookies_file:
        return sess, cookies_file
    else:
        return sess


def _parse_google_drive_file(url, content):
    """Extracts information about the current page file and its children."""

    folder_soup = bs4.BeautifulSoup(content, features="html.parser")

    # finds the script tag with window['_DRIVE_ivd']
    encoded_data = None
    for script in folder_soup.select("script"):
        inner_html = script.decode_contents()

        if "_DRIVE_ivd" in inner_html:
            # first js string is _DRIVE_ivd, the second one is the encoded arr
            regex_iter = re.compile(r"'((?:[^'\\]|\\.)*)'").finditer(
                inner_html
            )
            # get the second elem in the iter
            try:
                encoded_data = next(
                    itertools.islice(regex_iter, 1, None)
                ).group(1)
            except StopIteration:
                raise RuntimeError(
                    "Couldn't find the folder encoded JS string"
                )
            break

    if encoded_data is None:
        raise RuntimeError(
            "Cannot retrieve the folder information from the link. "
            "You may need to change the permission to "
            "'Anyone with the link'."
        )

    # decodes the array and evaluates it as a python array
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        decoded = encoded_data.encode("utf-8").decode("unicode_escape")
    folder_arr = json.loads(decoded)

    folder_contents = [] if folder_arr[0] is None else folder_arr[0]

    sep = " - "  # unicode dash
    splitted = folder_soup.title.contents[0].split(sep)
    if len(splitted) >= 2:
        name = sep.join(splitted[:-1])
    else:
        raise RuntimeError(
            "file/folder name cannot be extracted from: {}".format(
                folder_soup.title.contents[0]
            )
        )

    gdrive_file = _GoogleDriveFile(
        id=url.split("/")[-1],
        name=name,
        type=_GoogleDriveFile.TYPE_FOLDER,
    )

    id_name_type_iter = [
        (e[0], e[2].encode("raw_unicode_escape").decode("utf-8"), e[3])
        for e in folder_contents
    ]

    return gdrive_file, id_name_type_iter


def parse_google_drive_link(
    sess,
    url,
    quiet=False,
    remaining_ok=False,
    verify=True,
):
    """Get folder structure of Google Drive folder URL."""

    return_code = True

    # canonicalize the language into English
    if "?" in url:
        url += "&hl=en"
    else:
        url += "?hl=en"

    res = sess.get(url, verify=verify)

    if res.status_code != 200:
        return False, None

    gdrive_file, id_name_type_iter = _parse_google_drive_file(
        url=url,
        content=res.text,
    )

    return gdrive_file, id_name_type_iter


def check_and_download_one_folder(row):
    global root_path
    # Extract the folder's files in GDrive
    row = row[1]
    url = row['urls']
    #sess = _get_session(proxy=None, use_cookies=True)
    sess = _get_session(proxy=None, use_cookies=False)
    return_code, gdrive_file = parse_google_drive_link(sess, url, quiet=False, remaining_ok=True, verify=True)

    # Check if the files in GDrive are the same as the files in the folder
    output_folder_path = root_path / row['names']
    if output_folder_path.exists():
        listed_files = [f.name for f in output_folder_path.iterdir()]
        status = 0
        for g_file in gdrive_file:
            if g_file[1] not in listed_files:
                status += 1
        if status > 0:
            status = f'{status}/{len(gdrive_file)}'
        else:
            status = 'OK'        
    else:
        status = output_folder_path.name

    if status != 'OK':
        downloaded_files = gdown.download_folder(url=url, output=output_folder_path.as_posix(), quiet=True)
        if downloaded_files == None:
            status = f'{output_folder_path.name}: NOT OK after download'
        else:
            if len(gdrive_file) == len(downloaded_files):
                status = f'{output_folder_path.name}: OK after download'
            else:
                status = f'{output_folder_path.name}: {len(downloaded_files)}/{len(gdrive_file)} files downloaded. NOT OK'
    else:
        status = f'{output_folder_path.name}: OK'
    print(status)
    return status   


def generate_one_csv(url: str, csv_path: str):
    sess = _get_session(proxy=None, use_cookies=False)
    return_code, gdrive_root_folder = parse_google_drive_link(sess, url, quiet=False, remaining_ok=True, verify=True)
    df = {'names': [], 'urls': []}
    for g_folder in gdrive_root_folder:
        df['names'].append(g_folder[1])
        df['urls'].append(f'https://drive.google.com/drive/folders/{g_folder[0]}')
    df = pd.DataFrame(df)
    df.to_csv(csv_path, header=False, index=False)
    


def download_or_check_oak_multiprocessing(split:str, option: str, workers, generate_csvs=True):
    global root_path

    images_path = 'Raw'
    labels_path = 'Labels'

    root_path = Path('/home/tri110414/nfs_home/datasets/OAK/train')

    if option == 'images':
        root_path = root_path / images_path
        df = pd.read_csv('/home/tri110414/nfs_home/yolo-pruebas/datasets_utils/oak/train_images_folder.csv', names=['names', 'urls'], header=None)

    elif option == 'labels':
        root_path = root_path / labels_path
        df = pd.read_csv('/home/tri110414/nfs_home/yolo-pruebas/datasets_utils/oak/train_labels_folder.csv', names=['names', 'urls'], header=None)

    else:
        raise ValueError('Option must be images or labels')

    try:
        with Pool(workers) as p:
            folders_not_created = p.map(check_and_download_one_folder, df.iterrows())
    except json.decoder.JSONDecodeError:
        print('-'*60)
        print('JSONDecodeError catched. Downloads have not finished yet!')
        print('Restart program manually!'.upper())
        print('Exiting in 3 seconds...')
        print('-'*60)
        time.sleep(3)
        sys.exit()

    print(f'SUCCESFULLY FINISHED ALL DOWNLOADS OF {option} of {split}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='OAK Dataset downloader', 
        description='Downloads the OAK dataset. It must be restarted until message "SUCCESFULLY FINISHED ALL DOWNLOADS" appears',
    )
    #parser.add_argument('-s', '--split', type=str, help='train, val or test')
    parser.add_argument('-opt', '--option', type=str, help='images or labels')
    parser.add_argument('-w', '--workers', type=int, default=64, help='images or labels')
    #parser.add_argument('--only_check', action='store_true', help='only check if the files are downloaded')
    #parser.add_argument('-ow', '--overwrite_folders', action='store_true', help='images or labels')
    args = parser.parse_args()

    # overwrite_folders = args.overwrite_folders
    download_or_check_oak_multiprocessing(args.split, args.option, args.workers)
    


# def download_oak_images():

#     df = pd.read_csv('/home/tri110414/nfs_home/yolo-pruebas/datasets_utils/oak/images_folder.csv', names=['names', 'urls'], header=None)

#     root_path = Path('/home/tri110414/nfs_home/datasets/OAK/Raw')	# Path to the dataset imgs
#     root_path.mkdir(parents=False, exist_ok=True)

#     for i, row in df.iterrows():
#         output_folder_path = root_path / row['names']
#         output_folder_path.mkdir(parents=False, exist_ok=True)
#         url = row['urls']
#         gdown.download_folder(url=url, output=output_folder_path.as_posix(), quiet=True)
#         print(f'Downloaded {i+1}/{len(df)}')


# def download_oak_labels():
#     df = pd.read_csv('/home/tri110414/nfs_home/yolo-pruebas/datasets_utils/oak/labels_folder.csv', names=['names', 'urls'], header=None)

#     root_path = Path('/home/tri110414/nfs_home/datasets/OAK/Labels')	# Path to the dataset imgs
#     root_path.mkdir(parents=False, exist_ok=True)

#     for i, row in df.iterrows():
#         output_folder_path = root_path / row['names']
#         output_folder_path.mkdir(parents=False, exist_ok=True)

#         url = row['urls']
#         gdown.download_folder(url=url, output=output_folder_path.as_posix(), quiet=False)
#         print(f'Downloaded {i+1}/{len(df)}')

# def check_all_downloaded():

#     df = pd.read_csv('/home/tri110414/nfs_home/yolo-pruebas/datasets_utils/oak/images_folder.csv', names=['names', 'urls'], header=None)


#     for idx_folder, row in df.iterrows():

#         # Extract the folder's files in GDrive
#         url = row['urls']
#         #sess = _get_session(proxy=None, use_cookies=True)
#         sess = _get_session(proxy=None, use_cookies=False)
#         return_code, gdrive_file = parse_google_drive_link(sess, url, quiet=False, remaining_ok=True, verify=True)

#         # Check if the files in GDrive are the same as the files in the folder
#         output_folder_path = Path('/home/tri110414/nfs_home/datasets/OAK/Raw') / row['names']
#         if output_folder_path.exists():
#             listed_files = [f.name for f in output_folder_path.iterdir()]

#             for g_file in gdrive_file:
#                 if g_file[1] not in listed_files:
#                     print(f'File {g_file[1]} is not in {output_folder_path}')
#         else:
#             print(f'Folder {output_folder_path} does not exist')


# def download_one_folder_multiprocessing(row):
#     global root_path
#     url = row[1]['urls']
#     output_folder_path = root_path / row[1]['names']
#     if output_folder_path.exists() and not overwrite_folders:
#         print(f'{output_folder_path} already exists. Skipping')
#     else:
#         gdown.download_folder(url=url, output=output_folder_path.as_posix(), quiet=True)
#         print(f'Downloaded {row[0]+1}')