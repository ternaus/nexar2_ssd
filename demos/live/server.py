""" server.py

Ellis Brown, Max deGroot
"""

import sys
import socket

from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM, SOCK_STREAM
PORT_NUMBER = 5000
SIZE = 1024

hostName = gethostbyname('0.0.0.0')

s = socket(AF_INET, SOCK_STREAM)
s.bind((hostName, PORT_NUMBER))
s.listen(1)

conn, addr = s.accept()
# print("Test server listening on port {0}\n".format(PORT_NUMBER))
# print("Connection address: ", addr)

while True:
        data = s.recv(SIZE)
        if not data:
            break
        print("data received: ",  data)
        conn.send(data)

sys.exit()
