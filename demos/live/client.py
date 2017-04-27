""" client.py

Ellis Brown, Max deGroot
"""

import sys
from socket import socket, AF_INET, SOCK_DGRAM, SOCK_STREAM
import cv2
import numpy as np
import torch
import time
import struct

from cache import server_ip, parse_args

args = parse_args()

SERVER_IP = str(server_ip(args.ip))
print("SERVER_IP:", SERVER_IP)
PORT_NUMBER = 5000
SIZE = 3200
print("Test client sending packets to IP {0}, via port {1}\n"
      .format(SERVER_IP, PORT_NUMBER))

s = socket(AF_INET, SOCK_STREAM)
s.connect((SERVER_IP, PORT_NUMBER))

video_capture = cv2.VideoCapture(0)
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # data = pickle.dumps(frame)
    # s.sendall(struct.pack("L", len(data))+data)
    ret, jpeg = cv2.imencode('.jpg', frame)
    # jpeg = jpeg.tobytes()
    # s.send((b'--frame\r\n'
    # 	    b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r1\n'))
    try:
        s.send(jpeg.tostring())
        data = s.recv(SIZE)
        i = cv2.imdecode(np.fromstring(
            data, dtype=np.float32), cv2.IMREAD_COLOR)
        print(i)
    except ConnectionResetError:
        print('connection reset')
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('Video', i)
    cv2.waitKey(1)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
s.close()
sys.exit()
