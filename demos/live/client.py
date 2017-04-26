""" client.py

Ellis Brown, Max deGroot
"""

import sys
from socket import socket, AF_INET, SOCK_DGRAM, SOCK_STREAM

SERVER_IP = ''
PORT_NUMBER = 5000
SIZE = 1024
print("Test client sending packets to IP {0}, via port {1}\n"
      .format(SERVER_IP, PORT_NUMBER))

s = socket(AF_INET, SOCK_STREAM)
s.connect((SERVER_IP, PORT_NUMBER))

message = "cool"

while True:
    try:
        s.send(message.encode())
        data = s.recv(SIZE).decode()
    except ConnectionResetError:
        print('connection reset')
    print(data)
s.close()
sys.exit()

# while True:
#         mySocket.sendto(message.encode(), (SERVER_IP, PORT_NUMBER))
# sys.exit()
