# CT-SAN

This project hosts the code for our **CVPR 2017** paper.

this is tensorflow implementation of [End-to-end Concept word Detection for Video Captioning, Retrieval, and Question Answering](https://arxiv.org/abs/1610.02947) which proposes a high-level concept word detector that can be integrated with any video-to-language models. It takes a video as input and generates a list of concept words as useful semantic priors for language generation models.



## Reference

If you use this code or dataset as part of any published research, please refer the following paper.

```
@inproceedings{CT-SAN:2017:CVPR,
	author		= {Youngjae Yu and Hyungjin Ko and Jongwook Choi and Gunhee Kim},
	title		= {End-to-end Concept word Detection for Video Captioning, Retrieval, and Question Answering},
	booktitle	= {CVPR},
	year		= 2017
}
```



## Setup

### Get our code

```
git clone https://gitlab.com/fodrh1201/CT-SAN
```



### Basic Dependencies

```
pip install -r requirements.txt
```



### Python PATH

```
git submodule update --init --recursive
add2virtualenv .
add2 virtualenv coco-caption

cd coco/PythonAPI
python setup.py install
```



### Prepare Data

- Video Feature

  1. Download [LSMDC data](https://sites.google.com/site/describingmovies/lsmdc-2016/download).

  2. Extract all frames in videos into a separate folder. Here is one example script that extracts avi files into frames. You can save following script and run it with "./SCRIPT_NAME.sh INPUT_FOLDER avi OUTPUT_FOLDER"

     ```
     #!/bin/bash
     if [ "$1" == '' ] || [ "$2" == '' ] || [ "$3" == '' ]; then
     	echo "Usage: $0 <input folder> <file extension> <output folder>";
     	exit;
     fi
     for file in "$1"/*."$3"; do
     	destination="$2${file:${#1}:${#file}-${#1}-${#3}-1}";
     	mkdir -p "$destination";
     	ffmpeg -i "$file" "$destination/%d.jpg";
     done
     ```

  3. Extract ResNet-152 features by using each pretrained models

     - Extract 'res5c' for ResNet-152.
     - Only use one frame every five frames.

  4. Wrap each extracted features into hdf5 file, name as 'RESNET.hdf5' and save it in 'root/dataset/LSMDC/LSMDC16_features'. 

- Data frames
  - We processed raw data frames file in LSMDC16.
  - [[Download dataframes]](https://drive.google.com/open?id=0B1VtBNgsMJBgLXRseVhxVDhfSEE)
  - Save these files in "root/dataset/LSMDC/DataFrame"

- Vocabulary

  - We make word embedding matrix using GloVe Vector.
  - [Download vocabulary files](https://drive.google.com/open?id=0B1VtBNgsMJBga09ubXE4ajhGNjg)
  - These files include word embedding matrix file, word-index mapping file, and concept-index mapping file.
  - Save these files in "root/dataset/LSMDC/Vocabulary"
    ​



### Training

modify `configuartion.py` to suit your environment.

Run `train.py`.

```
python train.py
```

