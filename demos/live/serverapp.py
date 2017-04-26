import cv2
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot
import torch.utils.data as data
from PIL import Image
import sys
import os
from data import AnnotationTransform, VOCDetection, test_transform
from ssd import build_ssd
from timeit import default_timer as timer
from data import VOC_CLASSES as labelmap
import numpy as np
import urllib.request


stream = urllib.request.urlopen("XXXXX")
bytes=''
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

    app = Flask(__name__)

    @app.route('/')
    def index():
    	return render_template('index.html')

    def gen():
    	while True:
            bytes+=stream.read(1024)
            a = bytes.find('\xff\xd8')
            b = bytes.find('\xff\xd9')
            if a!=-1 and b!=-1:
                count+=1
                jpg = bytes[a:b+2]
                bytes= bytes[b+2:]
                i  = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
                if count % 100 == 0:
                    res = predict(i)
                    color = (0, 255, 0)
                    cv2.rectangle(image, (res[0], res[1]), (res[2]-res[0]+1, res[3]-res[1]+1), color, 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, res[4], (res[0], res[1]), font, 4,(255,255,255),2,cv2.LINE_AA)
                    ret, jpeg = cv2.imencode('.jpg',image)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r1\n')



    @app.route('/video_feed')
    def video_final():
    	return Response(gen(),
    		mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run(host='localhost', port = 8888, debug=True, threaded=True)
