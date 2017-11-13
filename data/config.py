# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4


# SSD300 and SSD512 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
v2 = {
    '300': {
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        # 'aspect_ratios': [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
        #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'v2',
    },
    # Based on https://github.com/balancap/SSD-Tensorflow/blob/9bfb15dd47c0140f97474597e9831e1583759ca9/nets/ssd_vgg_512.py
    '512': {
        'feature_maps': [64, 32, 16, 8, 6, 4, 2],
        'min_dim': 512,
        'steps': [8, 16, 32, 64, 128, 256, 512],
        'min_sizes': [20, 51, 133, 215, 297, 379, 461],
        'max_sizes': [51, 133, 215, 297, 379, 461, 543],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'v2',
    },
}

# use average pooling layer as last layer before multibox layers
v1 = {
    '300': {
        'feature_maps' : [38, 19, 10, 5, 3, 1],
        'min_dim' : 300,
        'steps' : [8, 16, 32, 64, 100, 300],
        'min_sizes' : [30, 60, 114, 168, 222, 276],
        'max_sizes' : [-1, 114, 168, 222, 276, 330],
        # 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        'aspect_ratios' : [[1,1,2,1/2],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],
                            [1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3]],
        'variance' : [0.1, 0.2],
        'clip' : True,
        'name' : 'v1',
    }
}
