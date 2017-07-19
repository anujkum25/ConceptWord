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

### Prerequisites

Make sure you have the following software installed on your system:

- Python 2.7
- Tensorflow 1.0+

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
  - Make video-file divided into frames using **ffmpeg** with 24 frames per second.
  - Use one frame every 5 five frames.
  - Extract Resnet-152 res5c feature and make it hdf file.
  - Make soft link in dataset folder(make datset folder in root)
- Data frames
  - we process raw data frames file in LSMDC16.
  - [[Download dataframes]](https://drive.google.com/open?id=0B1VtBNgsMJBgLXRseVhxVDhfSEE)
- Vocabulary
  - Embed words by GloVe word embedding



### Training

modify `configuartion.py` to suit your environment.

Run `train.py`.

```
python train.py
```

