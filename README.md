<div align="center">

<h1><a href="https://ieeexplore.ieee.org/document/11270006">A Hyperspectral Change Detection Method for Small Vehicles</a></h1>

Shuyi Xu, He Sun*✉, Xu Sun, Li Ni, and Lianru Gao

</div>

## Abstract
Small vehicles (SV) detection is crucial for urban security and traffic management. However, detecting such targets from a single image presents significant challenges due to the difficulty in discerning their dynamic movements. In this paper, we propose a deep joint image-level and feature-level processing network, IFNet, designed for detecting changes in SV using bi-temporal hyperspectral images. At the image-level, a new Gumbel Softmax trick (GS)-based band selection strategy is introduced to address the problem of inconsistent spectral resolutions of bi-temporal images. At the feature-level, to tackle the challenge of capturing edge and shape details of SV, we propose a feature-based edge enhancement module, it can extract the target edge using high-level difference features, and the refined change map will be generated with the guidance of the edge map. Moreover, current deep learning-based hyperspectral change detection (HCD) methods are limited by HCD datasets. Therefore, we propose a benchmark dataset, the Hyperspectral Vehicle Change Detection (HVCD) dataset, which consists of 201 pairs of aerial hyperspectral images, each with a size of 256×256 , and exhibits inconsistent spectral resolutions across the bi-temporal data. Extensive experiments conducted on the HVCD dataset demonstrate that our IFNet obtains state-of-the-art performance with an acceptable computational cost.

## HVCD Dataset 
Some example samples of HVCD dataset.
<img width="1280" height="622" alt="HVCD" src="https://github.com/user-attachments/assets/067f98e6-13f9-407a-94c7-87f4d9f2ce11" />

**Download [Link](https://pan.quark.cn/s/534a01f81e73?pwd=WbB1)**

## IFNet
Here, we provide the pytorch implementation of the paper: "A Hyperspectral Change Detection Method for Small Vehicles". 

For more information, please see our published paper in [[IEEE](https://ieeexplore.ieee.org/document/11270006)]  ***(Accepted by TIP 2025)***

![HCD-SMT](https://github.com/user-attachments/assets/51595399-d4d8-4a73-97ca-e673901c7e9f)

### Requirements
```python
Python 3.6
pytorch 1.6.0
torchvision 0.7.0
einops  0.3.0
```

### Train
Make sure you performed the data preparation above. Then, start training as follows:
```python
python train.py --model_name=IFNet --dataset=HVCD --gpu_ids=0 --batch_size=8 --epoches=200 
```

### Test
```python
python test.py --model_name=IFNet --dataset=HVCD --gpu_ids=0 --batch_size=8 --epoches=200 
```

## Citation: 
```
@ARTICLE{11270006,
  author={Xu, Shuyi and Sun, He and Sun, Xu and Ni, Li and Gao, Lianru},
  journal={IEEE Transactions on Image Processing}, 
  title={A Hyperspectral Change Detection Method for Small Vehicles}, 
  year={2025},
  volume={34},
  number={},
  pages={7874-7888},
  keywords={Feature extraction;Training;Image edge detection;Hyperspectral imaging;Spatial resolution;Sensors;Shape;Rivers;Vectors;Correlation;Hyperspectral change detection;small vehicles;band selection;transformer},
  doi={10.1109/TIP.2025.3635479}}
```
## Reference:
Thanks to the following repository: [BIT-CD](https://github.com/justchenhao/BIT_CD).
