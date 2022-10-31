# Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation
<div style="text-align : center;">
     <img src="assets/teaser.gif" width="1000">
</div>

<h2 align="center">ECCV 2022</h2>
<p align="center">
    <a href="https://eadcat.github.io/"><strong>Dae-Young Song</strong></a><sup>1</sup>
    ·
    <a href="https://www.geonsoo-lee.com/"><strong>Geonsoo Lee</strong></a><sup>1</sup>
    ·
    HeeKyung Lee<sup>2</sup>
    ·
    Gi-Mun Um<sup>2</sup>
    ·
    <a href="https://sites.google.com/view/cnu-cvip"><strong>Donghyeon Cho</strong></a><sup>1</sup>
</p>

<p align="center">
    <sup>1</sup> Chungnam National University
    ·
    <sup>2</sup> Electronics and Telecommunications Research Institute (ETRI)
</p>

An Official PyTorch Implementation of [Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760052.pdf). 

### INSTALLATION
```
conda create -n wssn python=3.8
conda activate wssn
sh env.sh
```

### DOWNLOAD
```
sh download.sh
```

or download the files below manually.

Dataset for Demo: [Google Drive (651M)](https://drive.google.com/file/d/1p27k77TWjknBYJ62EW97D2Xf_nElNZW3/view?usp=sharing)

Network Checkpoints: [Google Drive (3.9G)](https://drive.google.com/file/d/1AZr8eQa2m3fBkbb9t8MsWt-inbNwVVez/view?usp=sharing)

### DEMO
```
# If you want to use your own GPU, set the following options:
# Example 1: --gpu 0 1 2 3 --world-size 4 --npgpu 4  (DDP)
# Example 2: --gpu 2 3 --world-size 2 --npgpu 2  (DDP)
# Example 3: --gpu 0  (Single GPU)
# Check "options.py" for more details.

# All models below use single homography for one input.

# 01. Final Model
sh scripts/test-final.sh

# 02. Global Homography Only (W/O Local Adj.)
sh scripts/test-homography.sh

# 03. Without Color Correction
(https://ieeexplore.ieee.org/document/9393563)
sh scripts/test-spl.sh

# 04. Pre-color Correction (W/O Post-Color Correction)
sh scripts/test-pre.sh

# 05. Post-color Correction (W/O Pre-Color Correction)
sh scripts/test-post.sh

# 06. Final Model Trained with L1(Pixel-wise) Loss
sh scripts/test-L1.sh
```

### CITATION
```
@InProceedings{Song2022Weakly,
    author={Song, Dae-Young and Lee, Geonsoo and Lee, HeeKyung and Um, Gi-Mun and Cho, Donghyeon},
    title={Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation},
    journal={European Conference on Computer Vision (ECCV)},
    pages={54--71},
    year={2022},
    organization={Springer}
}

@article{song2021end,
  title={End-to-End Image Stitching Network via Multi-Homography Estimation},
  author={Song, Dae-Young and Um, Gi-Mun and Lee, Hee Kyung and Cho, Donghyeon},
  journal={IEEE Signal Processing Letters (SPL)},
  volume={28},
  pages={763--767},
  year={2021},
  publisher={IEEE}
}
```

### LICENSE
Data dual License -  CC BY-NC-ND 4.0, Commercial License

Source dual License - BSD-3-Clause License, Commercial License

### CONTACT
Question: eadyoung@naver.com; eadgaudiyoung@gmail.com

License: lhk95@etri.re.kr

If you want to use and/or redistribute this source commercially, please consult lhk95@etri.re.kr for details in advance.
