# Composite Binary Decomposition Networks
In this repository, we released code for Composite Binary Decomposition Networks (CBDNet).

### Contents
1. [Installation](#installation)
2. [Usage](#channel-pruning) 
3. [Experiment Result](#experiment-results) 
4. [Reference](#reference)

### Installation
1. Clone the repository of Caffe and compile it
```Shell
    git clone https://github.com/BVLC/caffe.git
    cd caffe
    # modify Makefile.config to the path of the library on your machine, please make sure the python interface is supported
    make -j8
    make pycaffe
```
2. Clone this repository 
```Shell
    https://github.com/wangzheng17/CBDNet.git
```
    
### Usage  
1. Download the original model files (.prototxt and .caffemodel) and move them to the directory of `models`

2. Command Line Usage
To decompose a network, use the following command
```Shell
    python main.py <arguments>
    arguments for CBDNet:
        -bottleneck           bottleneck ratio value (Numbers like 0.2,0.3,0.4,0.5,...)
        -j                    the number of overall channel except the sign channel(overall-fix channel, post-variable channel. Numbers like 5,6,7,8,...)
        -model MODEL          caffe prototxt file path
        -weight WEIGHT        caffemodel file path


```

For example, suppose the Resnet-18 network is in folder `models/` and named as `resnet-18.prototxt` and `resnet-18.caffemodel`, you can decompose all layers with same number of channels:
```Shell
    python main.py -model models/resnet-18.prototxt -weight models/resnet-18.caffemodel -bottleneck 0.5 -j 6
```

Note: please use same prefix name for prototxt and weights file, a floder will be created in "\models\". `btn_xx_a_list.npy` will be stored for reusing, where `xx` is the bottleneck ratio. Results are saved under `j_model` sub-directory for same number of non-compressed channel setting and same number of overall channel setting. The parameter 'j' in the result indicates overall channels include the sign.
Using model name compatiable with that in `line 33-43, rank_a.py`. These lines claim exclusive layers that don't require computing for specific model.

### Reference

This work is based on our work *Composite Binary Decomposition Networks (AAAI2019)*[[Paper]](https://arxiv.org/abs/1811.06668.pdf). If you think this is helpful for your research, please consider append following bibtex config in your latex file.

```Latex
```
