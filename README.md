<h1 align="center">🙈📐 Delaunay Triangulation Matching (DTM) 🖼️👀</h1>

<p align="center">
  <h2 align="center"><p>
    🔥 🐍 Python implementation </a> 🐍 🔥
  </p></h2>

<p align="center">
    <img src="https://github.com/fb82/DTM/blob/main/data/out/ET.jpg" alt="example" height=200>
    <img src="https://github.com/fb82/DTM/blob/main/data/out/DC.jpg" alt="example" height=200>
    <br>
    <em>HarrisZ+ and DoG + AffNet + OriNet + HardNet + <b>Blob matching</b> + <b>DTM</b> + RANSAC</em>
</p>

## What is it?
+ DTM is a non-deep spatial matching filter based on Delaunay triangulation, like [GMS](https://github.com/JiawangBian/GMS-Feature-Matcher) or [LPM](https://github.com/jiayi-ma/LPM?tab=readme-ov-file).
+ The [original code](https://sites.google.com/view/fbellavia/research/blob_dtm) was released in Matlab, this is the Python implementation.
+ DTM was originally developed to work with Blob matching, a general framework wich also includes matching strategies like MNN or NNR. Blob matching is also included in this implementation (but it is not mandatory).
+ For the best result, DTM requires as input the candidate matches and their similarity, but can works with spatial information only and no descriptor similarity clues.
+ DTM can also be used as guided matching to increase the correspondences. 

## Setup (including all the stuff to launch the demo)
Run from the terminal
```
git clone https://github.com/fb82/DTM.git
cd DTM
git submodule update --init --recursive
pip install -r requirements.txt
```
The current requirement file has been tested on Ubuntu 24.04. 

## Launch the demo
To use with the base image pair example run from the terminal 
```
python ./demo.py
```
or with your image pair as
```
python ./demo.py <path of 1st image> <path of the 2nd image>
```
For further details or customizations please inspect the comment in ``demo.py``.

## Notes
+ The demo use DoG and [HarrisZ+](https://github.com/fb82/HarrisZ) keypoints to compute the matches. In case of OOM issues you can select only one kind of keypoints.
+ Blob matching is implemented in PyTorch. By default it runs on CPU to avoid OOM, but if you have enough memory you can try on GPU. The demo allows alternatively the usage of the standard MNN.

## Where can I find more details?
See the paper [SIFT Matching by Context Exposed](https://arxiv.org/abs/2106.09584) (TPAMI 2022).
