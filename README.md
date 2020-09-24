# DLow: **D**iversifying **L**atent FL**ow**s
![Loading DLow Overview](data/overview.png "DLow Overview")
---
This repo contains the official implementation of our paper:
  
DLow: Diversifying Latent Flows for Diverse Human Motion Prediction  
Ye Yuan, Kris Kitani  
**ECCV 2020**  
[[website](https://www.ye-yuan.com/dlow)] [[paper](https://arxiv.org/pdf/2003.08386.pdf)] [[talk](https://youtu.be/c45ss6Tcb2A)] [[summary](https://youtu.be/nVYGHnRB1_M)] [[demo](https://youtu.be/64OEdSadb00)]

# Installation 
### Datasets
* Please follow the data preprocessing steps ([DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)) inside the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repo. Place the prepocessed data ``data_3d_h36m.npz`` (Human3.6M) and ``data_3d_humaneva15.npz`` (HumanEva-I) under the ``data`` folder.
### Environment
* **Tested OS:** MacOS, Linux
* **Packages:**
    * Python >= 3.6
    * [PyTorch](https://pytorch.org) >= 0.4
    * Tensorboard
* **Note**: All scripts should be run from the root of this repo to avoid path issues.

### Pretrained Models
* Download our pretrained models from [Google Drive](https://drive.google.com/file/d/1k5uDeUXrvtwZPN-lJNPSO8tPvHH6Gj55/view?usp=sharing) (or [BaiduYun](https://pan.baidu.com/s/1Ye6bHXcX6lNVMLaXJyzyWg), password: y9ph) and place the unzipped ``results`` folder inside the root of this repo.

# Train
### Configs
We have provided 4 example YAML configs inside ``motion_pred/cfg``: 
* `h36m_nsamp10.yml` and `h36m_nsamp50.yml` for Human3.6M for number of samples 10 and 50 respectively.
* `humaneva_nsamp10.yml` and `humaneva_nsamp50.yml` for HumanEva-I for number of samples 10 and 50 respectively.
* These configs also have corresponding pretrained models inside ``results``.

### Train VAE
```
python motion_pred/exp_vae.py --cfg h36m_nsamp10
```

### Train DLow (After VAE is trained)
```
python motion_pred/exp_dlow.py --cfg h36m_nsamp10
```

# Test 
### Visualize Motion Samples
```
python motion_pred/eval.py --cfg h36m_nsamp10 --mode vis
```
Useful keyboard shortcuts for the visualization GUI:  
| Key           | Functionality |
| ------------- | ------------- |
| d             | test next motion data
| c             | save current animation as `out/video.mp4` |
| space         | stop/resume animation |
| 1             | show DLow motion samples |
| 2             | show VAE motion samples |


### Compute Metrics
```
python motion_pred/eval.py --cfg h36m_nsamp50 --mode stats
```  


# Citation
If you find our work useful in your research, please cite our paper [DLow](https://www.ye-yuan.com/dlow):
```
@inproceedings{yuan2020dlow,
  title={Dlow: Diversifying latent flows for diverse human motion prediction},
  author={Yuan, Ye and Kitani, Kris},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

# Acknowledgement
Part of the code is borrowed from the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repo.

# License

The software in this repo is freely available for free non-commercial use. Please see the [license](LICENSE) for further details.
