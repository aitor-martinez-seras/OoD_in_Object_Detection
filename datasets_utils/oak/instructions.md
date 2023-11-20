# Instructions to download OAK

Instructions up to date in 14/11/2023. If paths or download site changes, refer to https://oakdata.github.io/ .

## Train

To download train files:
1. Change the hardcoded paths in oak_download.py
2. Run ```python download_oak.py -opt images -w W``` , where W is the desired number of processes.
3. Run it repeatedly until the message ```SUCCESFULLY FINISHED ALL DOWNLOADS``` appears.
4. Repeat 3 and 4 but pass the argument ```labels``` instead of ```images```.

## Validation

Manually download from https://drive.google.com/drive/folders/1_OkrQ35zUhjfcPnykYTe-YgDu17mB2SH and include the files in the datasets folder (outside the root folder of this project).