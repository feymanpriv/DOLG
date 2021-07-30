# DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features (ICCV 2021)

## Pipeline
<p align="center"><img width="90%" src="imgs/figure2.png" /></p>


## Performances
<p align="center"><img width="100%" src="imgs/result1.png" /></p>


## Codes

### Requirements

- NVIDIA GPU, Linux, Python3(tested on 3.6.10)
- Tested with CUDA 10.2, cuDNN 7.1 and PyTorch 1.4.0

```
pip install -r requirements.txt
```

### Training

1. Find datasets via symlinks from `datasets/data` to the actual locations where the dataset images and annotations are stored. Refer to [`DATA.md`](imgs/DATA.md).

2. Set datapath, model, training parameters in configs/example.yaml and run job.sh.


### Evaluation

1. Feature extraction, set ${total_num} = n * (gpu_cards) in configs/example.yaml and run evaler/run.sh for feature extraction. 

2. Eval on ROxf and RPar, refer [`README.md`](revisitop/README.md) for data fetch and description. Groudtruth file and some examples are prepared in [revisitop](https://github.com/feymanpriv/DOLG/tree/main/revisitop). 


## Wights
On the way ...


## Citation

If the project helps your research, please consider citing our paper as follows.

```BibTeX
@inproceedings{yang2021dolg,
  title = {DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features},
  author = {Min Yang and Dongliang He and Miao Fan and Baorong Shi and Xuetong Xue and Fu Li and Errui Ding and Jizhou Huang},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2021}
}

```


## References

pycls(https://github.com/facebookresearch/pycls)
pymetric(https://github.com/feymanpriv/pymetric)
DELG(https://github.com/feymanpriv/DELG)
Parsing-R-CNN(https://github.com/soeaver/Parsing-R-CNN)
