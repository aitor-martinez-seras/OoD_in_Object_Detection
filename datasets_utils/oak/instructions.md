# Instructions to download OAK

Instructions up to date in 14/11/2023. If paths or download site changes, refer to https://oakdata.github.io/ .

## Train

To download train images and labels:
1. Change the hardcoded paths in oak_download.py
2. Run ```python download_oak.py -opt images -w W``` , where W is the desired number of processes.
3. Run it repeatedly until the message ```SUCCESFULLY FINISHED ALL DOWNLOADS``` appears.
4. Repeat 3 and 4 but pass the argument ```labels``` instead of ```images```.

To download videos run ```python datasets_utils/oak/oak_download_videos.py --split train```

## Validation

Manually download from https://drive.google.com/drive/folders/1_OkrQ35zUhjfcPnykYTe-YgDu17mB2SH and include the files in the datasets folder (outside the root folder of this project).

To download videos run ```python datasets_utils/oak/oak_download_videos.py --split val```

# Instructions to generate .csv files with download URLs

Create a google sheet. Go to Extensions > Apps Script.

## .gs scripts

Now create the following .gs scripts:

1. Get train images and labels URLs. Choose one by commenting the other in the code:

```javascript
function myFunction() {
  var ss=SpreadsheetApp.getActiveSpreadsheet();
  var s=ss.getActiveSheet();
  var c=s.getActiveCell();

  // Images
  //var fldr=DriveApp.getFolderById("1t_XcuOcR_FYIR732bvB70FBW-IMZFVkW");
  //or
  // Labels
  var fldr=DriveApp.getFolderById("1uEDXnpd_uNo6MEUWo90-gYftE9rs4-Z4");
  
  // var files=fldr.getFiles();
  var folders=fldr.getFolders();
  var names=[],f,str;
  var urls =[],f,str;
  while (folders.hasNext()) {
    f=folders.next();
    // Logger.log(f)
    name = f.getName();
    url = f.getUrl();
    Logger.log(name)
    Logger.log(url)
    // str='=hyperlink("' + f.getUrl() + '","' + f.getName() + '")';
    names.push([name]);
    urls.push([url]);
  }
  //  s.getRange(c.getRow(),c.getColumn(),names.length).setFormulas(names);
  s.getRange(c.getRow(),c.getColumn(),names.length).setValues(names);
  s.getRange(c.getRow(),c.getColumn() + 1, urls.length).setValues(urls);
}
```

2. Get video URLs:

```javascript
function myFunction() {
  var ss=SpreadsheetApp.getActiveSpreadsheet();
  var s=ss.getActiveSheet();
  var c=s.getActiveCell();
  // Train videos
  var fldr=DriveApp.getFolderById("1RaC0zPyKOnNlAr_J8T25GsTSlm-lEu5n");
  var files=fldr.getFiles();
  //var folders=fldr.getFolders();
  var names=[],f,str;
  var urls =[],f,str;
  while (files.hasNext()) {
    f=files.next();
    Logger.log(f)
    name = f.getName();
    url = f.getUrl();
    Logger.log(name)
    Logger.log(url)
    // str='=hyperlink("' + f.getUrl() + '","' + f.getName() + '")';
    names.push([name]);
    urls.push([url]);
  }
  //  s.getRange(c.getRow(),c.getColumn(),names.length).setFormulas(names);
  s.getRange(c.getRow(),c.getColumn(),names.length).setValues(names);
  s.getRange(c.getRow(),c.getColumn() + 1, urls.length).setValues(urls);
}

```

## .csv files generation process

1. Execute the first script with the desired option selected in the code. In this example, first we select images (so commment the line of the Labels)
2. Check if in the opened sheet the info has been pasted
3. Then, click in ```Files > Download > .csv``` and the corresponding .csv file is downloaded
4. Remove the info from the sheet for future usages.
5. Change its name to the info it contains (train_images_folder.csv for example)
6. Modify the script by uncommenting the Labels variable and commenting the variable of the images 
7. Repeat steps 1 to 5
8. Execute the second script for the videos and make steps 2 to 5.


# Plots

