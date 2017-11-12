# SSD: Single Shot MultiBox Object Detector, in PyTorch for [NEXAR CHALLENGE II Vehicle Detection in the Wild using the NEXET Dataset](https://www.getnexar.com/challenge-2/)
A [PyTorch](http://pytorch.org/) implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C. Berg.  The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).  


Code is heavily based on (https://github.com/amdegroot/ssd.pytorch)


<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#demos'>Demos</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository.
  * Note: We currently only support Python 3+.
- Then download the dataset by following the [instructions](#download-voc2007-trainval--test) below.
- We now support [Visdom](https://github.com/facebookresearch/visdom) for real-time loss visualization during training! 
  * To use Visdom in the browser: 
  ```Shell
  # First install Python server and client 
  pip install visdom
  # Start the server (probably in a screen or tmux)
  python -m visdom.server
  ```
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).

## Training SSD
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python -m ssd_pytorch.train
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * Currently we only support training on v2 (the newest version).
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)
  
## Evaluation
To evaluate a trained network:

```Shell
python -m ssd_pytorch.eval
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  


<img align="left" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/detection_examples.png">


## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thank you to [Alex Koltun](https://github.com/alexkoltun) and his team at [Webyclip](webyclip.com) for their help in finishing the data augmentation portion.
- A list of other great SSD ports that were sources of inspiration (especially the Chainer repo): 
  * [Chainer](https://github.com/Hakuyume/chainer-ssd), [Keras](https://github.com/rykov8/ssd_keras), [MXNet](https://github.com/zhreshold/mxnet-ssd), [Tensorflow](https://github.com/balancap/SSD-Tensorflow) 
