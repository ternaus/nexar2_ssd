""" server.py

Ellis Brown, Max deGroot
"""

import sys
#import time
from socket import socket, gethostbyname, AF_INET, SOCK_STREAM
import torch
#import torch.nn as nn
import torch.backends.cudnn as cudnn
#import torchvision.transforms as transforms
from torch.autograd import Variable
#from sys import platform as sys_pf
# import torch.utils.data as data
import cv2
import numpy as np
#import struct
#import pickle

from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd

PORT_NUMBER = 5000
SIZE = 3200
hostName = gethostbyname('0.0.0.0')

trained_model = 'weights/ssd_300_VOC0712.pth'
net = build_ssd('test', 300, 21)    # initialize SSD
net.load_state_dict(torch.load(trained_model))
net = net.cuda()
cudnn.benchmark = True
net.eval()
transform = BaseTransform(net.size, (104, 117, 123))
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
font = cv2.FONT_HERSHEY_SIMPLEX


def predict(frame):
    height, width, _ = frame.shape
    x = Variable(transform(frame).unsqueeze(0).cuda(), volatile=True)
    y = net(x)  # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.4:
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                          int(pt[3])), colors[i % 3], 2)
            cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), font,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame


s = socket(AF_INET, SOCK_STREAM)
s.bind((hostName, PORT_NUMBER))
s.listen(2)
conn, addr = s.accept()
print("Test server listening on port {0}\n".format(PORT_NUMBER))
print("Connection address: ", addr)

data = ''
# payload_size = struct.calcsize("L")
while True:
    # while len(data) < payload_size:
    data = conn.recv(SIZE)
    # packed_msg_size = data[:payload_size]
    # data = data[payload_size:]
    # msg_size = struct.unpack("H", packed_msg_size)[0]
    # while len(data) < msg_size:
    #     data += conn.recv(4096)
    # frame_data = data[:msg_size]
    # data = data[msg_size:]
    # frame = pickle.loads(frame_data)
    i = cv2.imdecode(np.fromstring(data, dtype=np.float32), cv2.IMREAD_COLOR)
    if i is None:
        continue
    data = predict(i)
    print(data)
    ret, jpeg = cv2.imencode('.jpg', data)
    conn.send(jpeg.tostring())


sys.exit()
