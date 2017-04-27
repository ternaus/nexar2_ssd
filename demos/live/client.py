""" client.py

Ellis Brown, Max deGroot
"""

import sys
import socket
import cv2
import numpy as np
import time
from imutils.video import FPS, WebcamVideoStream


from cache import server_ip, parse_args

# networking
# args = parse_args()
# SERVER_IP = str(server_ip(args.ip))
# PORT = 5000
# SIZE = 3200
# print("Test client sending packets to IP {0}, via port {1}\n"
#       .format(SERVER_IP, PORT))
#
# sock = socket.socket(socket.AF_INET,     # Internet
#                      socket.SOCK_DGRAM)  # UDP
# sock.bind((SERVER_IP, PORT))

# start video stream thread, allow buffer to fill
print("[INFO] starting threaded video stream...")
stream = WebcamVideoStream(src=0).start()  # default camera
time.sleep(1.0)

# start fps timer
fps = FPS().start()

# loop over frames from the video file stream
while True:
    # grab next frame
    frame = stream.read()
    key = cv2.waitKey(1) & 0xFF

    # update FPS counter
    fps.update()

    # keybindings
    if key == ord('p'):  # pause
        while True:
            key2 = cv2.waitKey(1) or 0xff
            cv2.imshow('frame', frame)
            if key2 == ord('p'):  # resume
                break
    cv2.imshow('frame', frame)
    if key == 27:  # exit
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
stream.stop()
